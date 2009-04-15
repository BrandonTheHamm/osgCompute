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
#include "osgCompute/Computation"

#define COMPUTATIONBIN_NUMBER 1111
#define COMPUTATIONBIN_NAME "osgCompute::ComputationBin"

namespace osgCompute
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------ 
    Computation::Computation() 
        :   osg::Group() 
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------   
    void Computation::clearLocal()
    {
        _launchCallback = NULL;
        _modules.clear();
        _computeOrder = PRE_COMPUTE;
        _paramHandles.clear();
        _enabled = true;

        osg::Group::removeChildren(0,osg::Group::getNumChildren());
    }

    //------------------------------------------------------------------------------   
    void Computation::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Computation::accept(osg::NodeVisitor& nv) 
    { 
        if( nv.validNodeMask(*this) ) 
        {  
            nv.pushOntoNodePath(this);

            osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>( &nv );
            if( cv && _enabled )
            {
                addBin( *cv );
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
            {
                update( nv );
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
            {
                handleevent( nv );
            }

            traverse( nv );
            nv.popFromNodePath(); 
        } 
    }

    //------------------------------------------------------------------------------
    void Computation::loadModule( const std::string& modName )
    {
        // not implemented yet
    }
    
    //------------------------------------------------------------------------------
    void Computation::addModule( Module& module )
    {
        if( hasModule(module) )
            return;

        _modules.push_back( &module );

        checkCallbacks();

        for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            module.acceptParam( (*itr).first, *(*itr).second );
    }

    //------------------------------------------------------------------------------
    void Computation::removeModule( Module& module )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr) == &module )
            {
                _modules.erase( itr );
                checkCallbacks();
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->getName() == moduleName )
            {
                _modules.erase( itr );
                checkCallbacks();
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    bool Computation::hasModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Computation::hasModule( Module& module ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr) == &module )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Computation::hasModules() const 
    { 
        return !_modules.empty(); 
    }

    //-----------------------------------------------------------------------------
    Module* Computation::getModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //-----------------------------------------------------------------------------
    const Module* Computation::getModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    ModuleList* Computation::getModules() 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    const ModuleList* Computation::getModules() const 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    unsigned int Computation::getNumModules() const 
    { 
        return _modules.size(); 
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Computation::hasParamHandle( const std::string& handle ) const
    {
        HandleToParamMapCnstItr itr = _paramHandles.find( handle );
        if( itr != _paramHandles.end() )
            return true;

        return false;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::addParamHandle( const std::string& handle, Param& param )
    {
        _paramHandles.insert( std::make_pair< std::string, osg::ref_ptr<Param> >( handle, &param ) );
        checkCallbacks();

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->acceptParam( handle, param );
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::removeParamHandles( const std::string& handle )
    {
        bool found = false;
        HandleToParamMapItr itr = _paramHandles.find( handle );
        while( itr != _paramHandles.end() )
        {
            for( ModuleListItr modItr = _modules.begin(); modItr != _modules.end(); ++modItr )
                (*modItr)->removeParam( handle, (*itr).second.get() );

            _paramHandles.erase( itr );
            itr = _paramHandles.find( handle );
            found = true;
        }

        if( found )
            checkCallbacks();
    }

    //------------------------------------------------------------------------------
    void Computation::removeParamHandle( const osgCompute::Param& param )
    {
        for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
        {
            if( (*itr).second == &param )
            {
                for( ModuleListItr modItr = _modules.begin(); modItr != _modules.end(); ++modItr )
                    (*modItr)->removeParam( (*itr).first, (*itr).second.get() );

                _paramHandles.erase( itr );
                checkCallbacks();
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    HandleToParamMap* osgCompute::Computation::getParamHandles()
    {
        return &_paramHandles;
    }

    //------------------------------------------------------------------------------
    const HandleToParamMap* osgCompute::Computation::getParamHandles() const
    {
        return &_paramHandles;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Computation::addBin( osgUtil::CullVisitor& cv )
    {
        if( !cv.getState() )
        {
            osg::notify(osg::FATAL)  << "Computation::addBin() for \""
                << getName()<<"\": CullVisitor has no valid state."
                << std::endl;

            return;
        }

        osgUtil::RenderBin* curRB = cv.getCurrentRenderBin();
        if( !curRB )
        {
            osg::notify(osg::FATAL)  
                << "Computation::addBin() for \""<<getName()<<"\": current CullVisitor has no active RenderBin."
                << std::endl;

            return;
        }

        Context* ctx = getOrCreateContext( *cv.getState() );
        if( !ctx )
        {
            osg::notify(osg::FATAL)  
                << "Computation::addBin() for \""<<getName()<<"\": cannot create Context."
                << std::endl;

            return;
        }

        ///////////////////////
        // SETUP REDIRECTION //
        ///////////////////////
        unsigned int rbNum = 0;
        if( _computeOrder == POST_COMPUTE )
            rbNum = COMPUTATIONBIN_NUMBER;
        else
            rbNum = -COMPUTATIONBIN_NUMBER;

        ComputationBin* pb = 
            dynamic_cast<ComputationBin*>( curRB->find_or_insert(rbNum,COMPUTATIONBIN_NAME) );

        if( !pb )
        {
            osg::notify(osg::FATAL)  
                << "Computation::addBin() for \""<<getName()<<"\": cannot create ComputationBin."
                << std::endl;

            return;
        }

        pb->init( *this );
        pb->setContext( *ctx );
    }
    
    //------------------------------------------------------------------------------
    void Computation::update( osg::NodeVisitor& uv )
    {
        if( getUpdateCallback() )
            (*getUpdateCallback())( this, &uv );
        else
        {
            for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            {
                if( (*itr).second->getUpdateCallback() )
                    (*(*itr).second->getUpdateCallback())( *(*itr).second, uv );
            }

            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getUpdateCallback() )
                    (*(*itr)->getUpdateCallback())( *(*itr), uv );
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::handleevent( osg::NodeVisitor& ev )
    {
        if( getEventCallback() )
            (*getEventCallback())( this, &ev );
        else
        {
            for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            {
                if( (*itr).second->getEventCallback() )
                    (*(*itr).second->getEventCallback())( *(*itr).second, ev );
            }

            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getEventCallback() )
                    (*(*itr)->getEventCallback())( *(*itr), ev );
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::checkCallbacks()
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
    Context* Computation::getOrCreateContext( osg::State& state )
    {
        Context* context = osgCompute::Context::instance( state );
        if( context == NULL )
            context = osgCompute::Context::createInstance( state, contextLibraryName(), contextClassName() );

        // In case a object is not defined 
        // NULL will be returned
        return context;
    }    
    
    osgUtil::RegisterRenderBinProxy registerComputationBinProxy(COMPUTATIONBIN_NAME, new osgCompute::ComputationBin );
}
