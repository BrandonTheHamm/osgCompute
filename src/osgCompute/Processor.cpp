/* osgCompute - Copyright (C) 2008-2009 SVT Group
 *                                                                     
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 3 of
 * the License, or (at your option) any later version.
 *                                                                     
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesse General Public License for more details.
 *
 * The full license is in LICENSE file included with this distribution.
*/

#include <osg/NodeVisitor>
#include <osg/OperationThread>
#include <osgUtil/CullVisitor>
#include "osgCompute/Processor"

namespace osgCompute
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------ 
    Processor::Processor() 
        :   osg::Node() 
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------   
    void Processor::clearLocal()
    {
        _launchCallback = NULL;
        _modules.clear();
        _bins.clear();
        _paramHandles.clear();
    }

    //------------------------------------------------------------------------------   
    void Processor::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Processor::accept(osg::NodeVisitor& nv) 
    { 
        if( nv.validNodeMask(*this) ) 
        { 
            nv.pushOntoNodePath(this); 

            if( nv.getVisitorType() == osg::NodeVisitor::CULL_VISITOR )
            {
                osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>( &nv );
                if( !cv )
                {
                    osg::notify(osg::FATAL)  << "Processor::accept() for \""
                        << getName()<<"\": NodeVisitor is not a CullVisitor."
                        << std::endl;

                    return;
                }

                if( !cv->getState() )
                {
                    osg::notify(osg::FATAL)  << "Processor::accept() for \""
                        << getName()<<"\": CullVisitor has no valid state."
                        << std::endl;
                    return;
                }

                ///////////////////////
                // SETUP REDIRECTION //
                ///////////////////////
                Context* ctx = getOrCreateContext( *cv->getState() );
                ProcessorBin* bin = getOrCreateProcessorBin( nv );
                
                if( ctx && bin )
                {
                    
                    bin->setContext( *ctx );
                    cv->addDrawable( bin, NULL );
                }
                else
                {
                    osg::notify(osg::FATAL)  
                        << "Processor::accept(\"CULL_VISITOR\") for \""<<getName()<<"\": Redirection could not be created."
                        << std::endl;
                    return;
                }
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
            {
                update( nv );
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
            {
                handleevent( nv );
            }

            nv.popFromNodePath(); 
        } 
    }

    //------------------------------------------------------------------------------
    void Processor::loadModule( const std::string& modName )
    {
        // not implemented yet
    }
    
    //------------------------------------------------------------------------------
    void Processor::addModule( Module& module )
    {
        if( hasModule(module) )
            return;

        _modules.push_back( &module );

        checkTraversalModules();

        for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            module.acceptParam( (*itr).first, *(*itr).second );

        clearBins();
    }

    //------------------------------------------------------------------------------
    void Processor::removeModule( Module& module )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr) == &module )
            {
                _modules.erase( itr );
                checkTraversalModules();
                return;
            }
        }

        clearBins();
    }

    //------------------------------------------------------------------------------
    void Processor::removeModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->getName() == moduleName )
            {
                _modules.erase( itr );
                checkTraversalModules();
                return;
            }
        }

        clearBins();
    }

    //------------------------------------------------------------------------------
    bool Processor::hasModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Processor::hasModule( Module& module ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr) == &module )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Processor::hasModules() const 
    { 
        return !_modules.empty(); 
    }

    //-----------------------------------------------------------------------------
    Module* Processor::getModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //-----------------------------------------------------------------------------
    const Module* Processor::getModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    ModuleList* Processor::getModules() 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    const ModuleList* Processor::getModules() const 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    unsigned int Processor::getNumModules() const 
    { 
        return _modules.size(); 
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Processor::hasParamHandle( const std::string& handle ) const
    {
        HandleToParamMapCnstItr itr = _paramHandles.find( handle );
        if( itr != _paramHandles.end() )
            return true;

        return false;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Processor::addParamHandle( const std::string& handle, Param& param )
    {
        if( hasParamHandle( handle ) )
            return;

        _paramHandles.insert( std::make_pair< std::string, osg::ref_ptr<Param> >( handle, &param ) );

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->acceptParam( handle, param );

        clearBins();
    }

    //------------------------------------------------------------------------------
    void osgCompute::Processor::removeParamHandle( const std::string& handle )
    {
        HandleToParamMapItr itr = _paramHandles.find( handle );
        if( itr != _paramHandles.end() )
            _paramHandles.erase( itr );

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->removeParam( handle );

        clearBins();
    }

    //------------------------------------------------------------------------------
    HandleToParamMap* osgCompute::Processor::getParamHandles()
    {
        return &_paramHandles;
    }

    //------------------------------------------------------------------------------
    const HandleToParamMap* osgCompute::Processor::getParamHandles() const
    {
        return &_paramHandles;
    }

    //------------------------------------------------------------------------------
    void Processor::clearBins() 
    { 
        BinMapItr itr = _bins.begin();
        for(; itr != _bins.end(); ++itr) 
            if( (*itr).second.valid() )
                (*itr).second->clear(); 
    }

    //------------------------------------------------------------------------------
    void Processor::enableBins() 
    { 
        BinMapItr itr = _bins.begin();
        for(; itr != _bins.end(); ++itr) 
            if( (*itr).second.valid() )
                (*itr).second->enable(); 
    }

    //------------------------------------------------------------------------------
    void Processor::disableBins() 
    { 
        BinMapItr itr = _bins.begin();
        for(; itr != _bins.end(); ++itr) 
            if( (*itr).second.valid() )
                (*itr).second->disable(); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Processor::update( osg::NodeVisitor& uv )
    {
        if( getUpdateCallback() )
            (*getUpdateCallback())( this, &uv );
        else
        {
            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getUpdateCallback() )
                    (*itr)->getUpdateCallback()->update( *(*itr), uv );
            }

            for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            {
                if( (*itr).second->getUpdateCallback() )
                    (*itr).second->getUpdateCallback()->update( *(*itr).second, uv );
            }
        }
    }

    //------------------------------------------------------------------------------
    void Processor::handleevent( osg::NodeVisitor& ev )
    {
        if( getEventCallback() )
            (*getEventCallback())( this, &ev );
        else
        {
            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getEventCallback() )
                    (*itr)->getEventCallback()->handleevent( *(*itr), ev );
            }



            for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            {
                if( (*itr).second->getEventCallback() )
                    (*itr).second->getEventCallback()->handleevent( *(*itr).second, ev );
            }
        }
    }

    //------------------------------------------------------------------------------
    void Processor::checkTraversalModules()
    {
        unsigned int numUpdates = 0;
        unsigned int numEvents = 0;

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->getEventCallback() )
                numEvents++;

            if( (*itr)->getUpdateCallback() )
                numUpdates++;
        }

        for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
        {
            if( (*itr).second->getEventCallback() )
                numEvents++;

            if( (*itr).second->getUpdateCallback() )
                numUpdates++;
        }

        osg::Node::setNumChildrenRequiringUpdateTraversal( numUpdates );
        osg::Node::setNumChildrenRequiringEventTraversal( numEvents );
    }

    //------------------------------------------------------------------------------
    Context* Processor::getOrCreateContext( osg::State& state )
    {
        Context* context = osgCompute::Context::instance( state );
        if( context == NULL )
            context = osgCompute::Context::createInstance( state, contextLibraryName(), contextClassName() );

        // In case a object is not defined 
        // NULL will be returned
        return context;
    }

    //------------------------------------------------------------------------------
    ProcessorBin* Processor::getOrCreateProcessorBin( osg::NodeVisitor& nv )
    {
        // lock mutex to prevent distinct threads from
        // changing each others cache. Note that each 
        // thread must occupy a different object
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        ProcessorBin* pb = NULL;
        BinMapItr itr = _bins.find( &nv );
        if( itr == _bins.end() )
        {   
            pb = newProcessorBin();
            if( pb )
                _bins[&nv] = pb;
        }
        else
        {
            pb = (*itr).second.get();
        }

        if( pb->isDirty() )
            pb->init( *this );

        // In case a object is not defined 
        // NULL will be returned
        return pb;
    }
}
