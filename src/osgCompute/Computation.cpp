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

#include <sstream>
#include <osg/NodeVisitor>
#include <osg/OperationThread>
#include <osgUtil/CullVisitor>
#include <osgCompute/Visitor>
#include <osgCompute/Computation>

namespace osgCompute
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------ 
    Computation::Computation() 
        :   osg::Group(),
            _parentComputation( NULL )
    { 
        clearLocal(); 
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
            ResourceVisitor* rv = dynamic_cast<ResourceVisitor*>( &nv );
            if( (cv != NULL) && 
                _enabled )
            {
                setupBin( *cv );
            }
            else
            {
                if( rv != NULL )
                {
                    collectResources();
                }
                else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
                {
                    update( nv );
                }
                else if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
                {
                    handleevent( nv );
                }

                nv.apply( *this );
            }

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
        Resource* curResource = NULL;
        for( ResourceMapItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            curResource = (*itr).first;
            if( !curResource )
                continue;

            module.acceptResource( *curResource );
        }

        // increment traversal counter if required
        if( module.getEventResourceCallback() )
            osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() + 1 );

        if( module.getUpdateResourceCallback() )
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );

        _modules.push_back( &module );
        resourcesChanged();
    }

    //------------------------------------------------------------------------------
    void Computation::removeModule( Module& module )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr) == &module )
            {
                // decrement traversal counter if necessary
                if( module.getEventResourceCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( module.getUpdateResourceCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                _modules.erase( itr );
                resourcesChanged();
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeModule( const std::string& moduleHandle )
    {
        ModuleListItr itr = _modules.begin();
        while( itr != _modules.end() )
        {
            if( (*itr)->isAddressedByHandle( moduleHandle ) )
            {
                // decrement traversal counter if necessary
                if( (*itr)->getEventResourceCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( (*itr)->getUpdateResourceCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                _modules.erase( itr );
                itr = _modules.begin();
                resourcesChanged();
            }
            else
            {
                ++itr;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeModules()
    {
        ModuleListItr itr = _modules.begin();
        while( itr != _modules.end() )
        {
            Module* curModule = (*itr).get();
            if( curModule != NULL )
            {
                // decrement traversal counter if necessary
                if( curModule->getEventResourceCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( curModule->getUpdateResourceCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );
            }

            _modules.erase( itr );
            itr = _modules.begin();
        }

        resourcesChanged();
    }

    //------------------------------------------------------------------------------
    bool Computation::hasModule( const std::string& moduleHandle ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->isAddressedByHandle( moduleHandle ) )
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

    //------------------------------------------------------------------------------
    ModuleList& Computation::getModules() 
    { 
        return _modules; 
    }

    //------------------------------------------------------------------------------
    const ModuleList& Computation::getModules() const 
    { 
        return _modules; 
    }

    //------------------------------------------------------------------------------
    unsigned int Computation::getNumModules() const 
    { 
        return _modules.size(); 
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Computation::hasResource( Resource& resource ) const
    {
        ResourceMapCnstItr itr = _resources.find( &resource );
        if( itr != _resources.end() )
            return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Computation::hasResource( const std::string& handle ) const
    {
        Resource* curResource = NULL;
        for( ResourceMapCnstItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            curResource = (*itr).first;
            if( !curResource )
                continue;

            if( curResource->isAddressedByHandle(handle)  )
                return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::addResource( Resource& resource )
    {
        if( hasResource(resource) )
            return;

        // increment traversal counter if required
        if( resource.getEventResourceCallback() )
            osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() + 1 );

        if( resource.getUpdateResourceCallback() )
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->acceptResource( resource );

        _resources.insert( std::make_pair< Resource*, osg::ref_ptr<osg::Object> > ( &resource, resource.asObject() ) );
        resourcesChanged();
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::removeResource( const std::string& handle )
    {
        Resource* curResource = NULL;

        ResourceMapItr itr = _resources.begin();
        while( itr != _resources.end() )
        {
            curResource = (*itr).first;
            if( !curResource )
                continue;

            if( curResource->isAddressedByHandle( handle ) )
            {
                // decrement traversal counter if necessary
                if( curResource->getEventResourceCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( curResource->getUpdateResourceCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                    (*moditr)->removeResource( *curResource );

                _resources.erase( itr );
                itr = _resources.begin();
                resourcesChanged();
            }
            else
            {
                ++itr;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeResource( Resource& resource )
    {
        ResourceMapItr itr = _resources.find( &resource );
        if( itr != _resources.end() )
        {
            // decrement traversal counter if necessary
            if( resource.getEventResourceCallback() )
                osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

            if( resource.getUpdateResourceCallback() )
                osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

            for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                (*moditr)->removeResource( resource );

             _resources.erase( itr );
             resourcesChanged();
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeResources()
    {
        ResourceMapItr itr = _resources.begin();
        while( itr != _resources.end() )
        {
            Resource* curResource = (*itr).first;
            if( curResource != NULL )
            {
                // decrement traversal counter if necessary
                if( curResource->getEventResourceCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( curResource->getUpdateResourceCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                    (*moditr)->removeResource( *curResource );
            }

            _resources.erase( itr );
            itr = _resources.begin();
        }
        resourcesChanged();
    }

    //------------------------------------------------------------------------------
    ResourceMap& osgCompute::Computation::getResources()
    {
        return _resources;
    }

    //------------------------------------------------------------------------------
    const ResourceMap& osgCompute::Computation::getResources() const
    {
        return _resources;
    }

    //------------------------------------------------------------------------------
    void Computation::resourcesChanged()
    {
        // we have to check the dirty flag of all resources in the update traversal 
        // whenever a resource has been added and we need to collect
        // resources located in the sub-graph
        if( _resourcesChanged == false )
        {
            _resourcesChanged = true;
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );
        }

        // notify parent graph
        if( getParentComputation() )
            getParentComputation()->resourcesChanged();

        _resourcesCollected = false;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------   
    void Computation::clearLocal()
    {
        _resourcesChanged = false;
        _resourcesCollected = false;
        removeResources();
        removeModules();
        _computeOrder = PRE_COMPUTE_POST_TRAVERSAL;
        _launchCallback = NULL;
        _enabled = true;
        _autoCollectResources = false;
        _parentComputation = NULL;
        _resourceVisitor = NULL;
        _contextMap.clear();

        removeChildren(0,osg::Group::getNumChildren());
    }

    //------------------------------------------------------------------------------
    void Computation::setParentComputation( Computation* parentComputation )
    {
        // different parent so clear context map
        if( parentComputation != _parentComputation )
            _contextMap.clear();

        _parentComputation = parentComputation;
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Computation::setContext( Context& context )
    {
        _contextMap[context.getId()] = &context;
        return true;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::removeContext( unsigned int ctxId )
    {
        ContextMapItr itr = _contextMap.find( ctxId );
        if( itr != _contextMap.end() )
            _contextMap.erase(itr);
    }

    //------------------------------------------------------------------------------
    Context* Computation::getContext( unsigned int ctxId )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        ContextMapItr itr = _contextMap.find( ctxId );
        if( itr == _contextMap.end() )
            return NULL;

        return (*itr).second.get();
    }

    //------------------------------------------------------------------------------
    Context* Computation::getOrCreateContext( unsigned int ctxId )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        Context* context = NULL;
        ContextMapItr itr = _contextMap.find( ctxId );
        if( itr == _contextMap.end() )
        {
            context = newContext();
            if( !context )
            {
                osg::notify(osg::FATAL)  
                    << "Computation::getOrCreateContext() for \""<<getName()<<"\": cannot create context."
                    << std::endl;

                return NULL;
            }


            context->setId( ctxId );
            std::stringstream contextName;
            contextName << getName() <<"_Context_" << ctxId; 
            context->setName( contextName.str() );

            _contextMap.insert( std::make_pair< unsigned int, osg::ref_ptr<Context> >( ctxId, context) );
        }
        else
        {
            context = (*itr).second.get();
        }

        // In case a object is not defined 
        // NULL will be returned
        return context;
    }

    //------------------------------------------------------------------------------
    void Computation::setupBin( osgUtil::CullVisitor& cv )
    {
        if( !cv.getState() )
        {
            osg::notify(osg::FATAL)  << "Computation::addBin() for \""
                << getName()<<"\": CullVisitor must provide a valid state."
                << std::endl;

            return;
        }

        Context* ctx = getOrCreateContext( cv.getState()->getContextID() );
        if( !ctx )
        {
            osg::notify(osg::FATAL)  
                << "Computation::addBin() for \""<<getName()<<"\": cannot create Context."
                << std::endl;

            return;
        }
        if( ctx->getState() != cv.getState() )
            ctx->setState( *cv.getState() );

        if( NULL == getParentComputation() )
            distributeContext( *ctx );

        ///////////////////////
        // SETUP REDIRECTION //
        ///////////////////////
        osgUtil::RenderBin* oldRB = cv.getCurrentRenderBin();
        if( !oldRB )
        {
            osg::notify(osg::FATAL)  
                << "Computation::addBin() for \""<<getName()<<"\": current CullVisitor has no active RenderBin."
                << std::endl;

            return;
        }
        const osgUtil::RenderBin::RenderBinList& rbList = oldRB->getRenderBinList();

        // we have to look for a better method to add more computation bins
        // to the same hierachy level
        int rbNum = 0;
        if( (_computeOrder & POST_COMPUTE) != POST_COMPUTE )
        {
            osgUtil::RenderBin::RenderBinList::const_iterator itr = rbList.begin();
            if( itr != rbList.end() && (*itr).first < 0 )
                rbNum = (*itr).first - 1;
            else
                rbNum = -1;
        }
        else
        {
            osgUtil::RenderBin::RenderBinList::const_reverse_iterator ritr = rbList.rbegin();
            if( ritr != rbList.rend() && (*ritr).first > 0 )
                rbNum = (*ritr).first + 1;
            else
                rbNum = 1;
        }

        std::string compBinName;
        compBinName.append( binLibraryName() );
        compBinName.append( "::" );
        compBinName.append( binClassName() );

        ComputationBin* pb = 
            dynamic_cast<ComputationBin*>( oldRB->find_or_insert(rbNum,compBinName) );

        if( !pb )
        {
            osg::notify(osg::FATAL)  
                << "Computation::addBin() for \""<<getName()<<"\": cannot create ComputationBin."
                << std::endl;

            return;
        }

        pb->init( *this );
        pb->setContext( *ctx );

        cv.setCurrentRenderBin( pb );
        cv.apply( *this );
        cv.setCurrentRenderBin( oldRB );
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::distributeContext( Context& context )
    {
        osg::ref_ptr<ContextVisitor> ctxVisitor = new ContextVisitor;
        ctxVisitor->setContext( &context );
        if( !ctxVisitor->init() )
        {
            osg::notify(osg::FATAL)  
                << "Computation::distributeContext() for \""<<getName()<<"\": cannot init context visitor."
                << std::endl;

            return;
        }

        // distribute context to the subgraph
        ctxVisitor->apply( *this );
    }
    
    //------------------------------------------------------------------------------
    void Computation::update( osg::NodeVisitor& uv )
    {
        if( !_resourcesCollected && getParentComputation() != NULL )
        {
            // status changed from child node to topmost node 
            // so clear context list
            _contextMap.clear();
            setParentComputation( NULL );
        }

        if( _resourcesChanged )
        {
            // topmost node starts 
            // resource collection
            if( !_resourcesCollected )
                collectResources();

            // init params if not done so far 
            Resource* curResource = NULL;
            for( ResourceMapItr itr = _resources.begin(); itr != _resources.end(); ++itr )
            {
                curResource = (*itr).first;
                if( !curResource || curResource->asModule() != NULL )
                    continue;

                if( curResource->isDirty() )
                    curResource->init();
            }

            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->isDirty() )
                    (*itr)->init();
            }

            // decrement update counter when all resources have been initialized
            osg::Node::setNumChildrenRequiringUpdateTraversal( 
                osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

            // set resources changed to false
            _resourcesChanged = false;

            // if auto update is selected then we have to traverse the
            // sub-graph each update cycle
            if( getAutoCollectResources() )
                resourcesChanged();
        }

        if( getUpdateCallback() )
            (*getUpdateCallback())( this, &uv );
        else
        {
            Resource* curResource = NULL;
            for( ResourceMapItr itr = _resources.begin(); itr != _resources.end(); ++itr )
            {
                curResource = (*itr).first;
                if( !curResource || curResource->asModule() != NULL )
                    continue;
                if( curResource->getUpdateResourceCallback() )
                    (*curResource->getUpdateResourceCallback())( *curResource, uv );
            }

            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getUpdateResourceCallback() )
                    (*(*itr)->getUpdateResourceCallback())( *(*itr), uv );
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::collectResources()
    {
        if( !_resourceVisitor.valid() )
        {
            _resourceVisitor = new ResourceVisitor;
            _resourceVisitor->setComputation( this );
            if( !_resourceVisitor->init() )
            {
                osg::notify(osg::FATAL)  
                    << "Computation::collectResources() for \""<<getName()<<"\": cannot init resource visitor."
                    << std::endl;

                return;
            }
        }
        _resourceVisitor->setupForTraversal();

        // collect resources and setup parent computations 
        // in the subgraph
        _resourceVisitor->traverse( *this );

        // setup resources for this computation
        _resourceVisitor->updateComputation();
        _resourcesCollected = true;
    }

    //------------------------------------------------------------------------------
    void Computation::handleevent( osg::NodeVisitor& ev )
    {
        if( getEventCallback() )
            (*getEventCallback())( this, &ev );
        else
        {
            Resource* curResource = NULL;
            for( ResourceMapItr itr = _resources.begin(); itr != _resources.end(); ++itr )
            {
                curResource = (*itr).first;
                if( !curResource || curResource->asModule() != NULL )
                    continue;

                if( curResource->getEventResourceCallback() )
                    (*curResource->getEventResourceCallback())( *curResource, ev );
            }

            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getEventResourceCallback() )
                    (*(*itr)->getEventResourceCallback())( *(*itr), ev );
            }
        }
    }   
}
