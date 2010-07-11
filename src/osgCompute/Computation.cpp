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
#include <osgUtil/GLObjectsVisitor>
#include <osgCompute/Visitor>
#include <osgCompute/Computation>

namespace osgCompute
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    bool Computation::s_deviceReady = false;

    //------------------------------------------------------------------------------
    bool Computation::isDeviceReady()
    {
        return s_deviceReady;
    }

    //------------------------------------------------------------------------------
    void Computation::setDeviceReady()
    {
        s_deviceReady = true;
    }

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
        if( !isDeviceReady() )
            checkDevice();

        if( nv.validNodeMask(*this) ) 
        {  
            nv.pushOntoNodePath(this);

            osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>( &nv );
            osgUtil::GLObjectsVisitor* ov = dynamic_cast<osgUtil::GLObjectsVisitor*>( &nv );
            ResourceVisitor* rv = dynamic_cast<ResourceVisitor*>( &nv );
            if( cv != NULL )
            {
                if( _enabled && (_computeOrder & OSGCOMPUTE_RENDER) == OSGCOMPUTE_RENDER )
                    addBin( *cv );
                else
                    nv.apply(*this);
            }
            else if( ov != NULL )
            {
                addContext( *ov->getState() );

                // Setup state
                if( !isDeviceReady() ) 
                {               
                    osg::notify(osg::FATAL) 
                        << getName() << " [Computation::accept(GLObjectsVisitor)]: No valid GL Device found."
                        << std::endl;

                    return;
                }

                nv.apply( *this );
            }
            else if( rv != NULL )
            {
                collectResources();

                nv.apply( *this );
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
            {
                update( nv );

                if( _enabled && (_computeOrder & UPDATE_PRE_TRAVERSAL) == UPDATE_PRE_TRAVERSAL )
                    launch();

                nv.apply( *this );

                if( _enabled && (_computeOrder & UPDATE_POST_TRAVERSAL) == UPDATE_POST_TRAVERSAL )
                    launch();
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
            {
                handleevent( nv );

                nv.apply( *this );
            }
            else
            {
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
        for( ResourceHandleListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            curResource = (*itr)._resource.get();
            if( !curResource )
                continue;

            module.acceptResource( *curResource );
        }

        // increment traversal counter if required
        if( module.getEventCallback() )
            osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() + 1 );

        if( module.getUpdateCallback() )
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );


        _modules.push_back( &module );
        subgraphChanged();
    }

    //------------------------------------------------------------------------------
    void Computation::removeModule( Module& module )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr) == &module )
            {
                // decrement traversal counter if necessary
                if( module.getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( module.getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                _modules.erase( itr );
                subgraphChanged();
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeModule( const std::string& moduleIdentifier )
    {
        ModuleListItr itr = _modules.begin();
        while( itr != _modules.end() )
        {
            if( (*itr)->isIdentifiedBy( moduleIdentifier ) )
            {
                // decrement traversal counter if necessary
                if( (*itr)->getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( (*itr)->getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );


                _modules.erase( itr );
                itr = _modules.begin();
                subgraphChanged();
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
        ModuleListItr itr;
        while( !_modules.empty() )
        {
            itr = _modules.begin();

            Module* curModule = (*itr).get();
            if( curModule != NULL )
            {
                // decrement traversal counter if necessary
                if( curModule->getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( curModule->getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );
            }
            _modules.erase( itr );
        }

        subgraphChanged();
    }

    //------------------------------------------------------------------------------
    bool Computation::hasModule( const std::string& moduleIdentifier ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->isIdentifiedBy( moduleIdentifier ) )
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
		for( ResourceHandleListCnstItr itr = _resources.begin();
			itr != _resources.end();
			++itr )
		{
			if( (*itr)._resource == &resource )
				return true;
		}

        return false;
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Computation::hasResource( const std::string& handle ) const
    {
        Resource* curResource = NULL;
        for( ResourceHandleListCnstItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            if(	(*itr)._resource.valid() && (*itr)._resource->isIdentifiedBy(handle)  )
                return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::addResource( Resource& resource, bool attach )
    {
        if( hasResource(resource) )
            return;

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->acceptResource( resource );

		ResourceHandle newHandle;
		newHandle._resource = &resource;
		newHandle._attached = attach;
		_resources.push_back( newHandle );
        subgraphChanged();
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::removeResource( const std::string& handle )
    {
        Resource* curResource = NULL;

        ResourceHandleListItr itr = _resources.begin();
        while( itr != _resources.end() )
        {
            curResource = (*itr)._resource.get();
            if( !curResource )
                continue;

            if( curResource->isIdentifiedBy( handle ) )
            {
                for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                    (*moditr)->removeResource( *curResource );

                _resources.erase( itr );
                itr = _resources.begin();
                subgraphChanged();
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
		for( ResourceHandleListItr itr = _resources.begin();
			itr != _resources.end();
			++itr )
		{
			if( (*itr)._resource == &resource )
			{
				for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
					(*moditr)->removeResource( resource );

				_resources.erase( itr );
				subgraphChanged();
				return;
			}
		}

    }

    //------------------------------------------------------------------------------
    void Computation::removeResources()
    {
        ResourceHandleListItr itr = _resources.begin();
        while( itr != _resources.end() )
        {
            Resource* curResource = (*itr)._resource.get();
            if( curResource != NULL )
            {
                for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                    (*moditr)->removeResource( *curResource );
            }

            _resources.erase( itr );
            itr = _resources.begin();
        }
        subgraphChanged();
    }

    //------------------------------------------------------------------------------
    ResourceHandleList& Computation::getResources()
    {
        return _resources;
    }

    //------------------------------------------------------------------------------
    const ResourceHandleList& Computation::getResources() const
    {
        return _resources;
    }

	//------------------------------------------------------------------------------
	bool Computation::isResourceAttached( Resource& resource ) const
	{
		for( ResourceHandleListCnstItr itr = _resources.begin();
			itr != _resources.end();
			++itr )
		{
			if( (*itr)._resource == &resource )
				return (*itr)._attached;
		}

		return false;
	}


    //------------------------------------------------------------------------------
    void Computation::subgraphChanged()
    {
        // we have to check the dirty flag of all resources in the update traversal 
        // whenever a resource has been added and we need to collect
        // resources located in the sub-graph
        if( _subgraphChanged == false )
        {
            _subgraphChanged = true;
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );
        }

        // notify parent graph
        if( getParentComputation() )
            getParentComputation()->subgraphChanged();

        _resourcesCollected = false;
    }



    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Computation::setLaunchCallback( LaunchCallback* lc ) 
    { 
        if( lc == _launchCallback )
            return;

        _launchCallback = lc; 
    }

    //------------------------------------------------------------------------------
    LaunchCallback* Computation::getLaunchCallback() 
    { 
        return _launchCallback; 
    }

    //------------------------------------------------------------------------------
    const LaunchCallback* Computation::getLaunchCallback() const 
    { 
        return _launchCallback; 
    }

    //------------------------------------------------------------------------------
    void Computation::setComputeOrder( Computation::ComputeOrder co )
    {
        // deactivate auto update
        if( (_computeOrder & OSGCOMPUTE_UPDATE ) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( getNumChildrenRequiringUpdateTraversal() - 1 );

        _computeOrder = co;

        // set auto update active in case we use the update traversal to compute things
        if( (_computeOrder & OSGCOMPUTE_UPDATE ) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( getNumChildrenRequiringUpdateTraversal() + 1 );
    }

    //------------------------------------------------------------------------------
    Computation::ComputeOrder Computation::getComputeOrder() const
    {
        return _computeOrder;
    }

    //------------------------------------------------------------------------------
    Computation* Computation::getParentComputation() const
    {
        return _parentComputation;
    }

    //------------------------------------------------------------------------------
    void Computation::setAutoCheckSubgraph( bool autoCheckSubgraph )
    {
        _autoCheckSubgraph = autoCheckSubgraph;
    }

    //------------------------------------------------------------------------------
    bool Computation::getAutoCheckSubgraph()
    {
        return _autoCheckSubgraph;
    }

    //------------------------------------------------------------------------------
    void Computation::enable() 
    { 
        _enabled = true; 
    }

    //------------------------------------------------------------------------------
    void Computation::disable() 
    { 
        _enabled = false; 
    }

    //------------------------------------------------------------------------------
    bool Computation::isEnabled() const
    {
        return _enabled;
    }

    //------------------------------------------------------------------------------
    void Computation::releaseGLObjects( osg::State* state ) const
    {
        if( state != NULL )
        {
            ContextSetItr itr = _contextSet.find( state->getGraphicsContext() );
            if( itr != _contextSet.end() )
                _contextSet.erase( state->getGraphicsContext() );
            
            Group::releaseGLObjects( state );
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------   
    void Computation::clearLocal()
    {
        _subgraphChanged = false;
        _resourcesCollected = false;
        removeResources();
        removeModules();
        _launchCallback = NULL;
        _enabled = true;
        _autoCheckSubgraph = false;
        _parentComputation = NULL;
        _resourceVisitor = NULL;
        _contextSet.clear();

        // clear node or group related members
        removeChildren(0,osg::Group::getNumChildren());
        setDataVariance( osg::Object::DYNAMIC );
        setUpdateCallback( NULL );
        setEventCallback( NULL );

        // setup computation order
        _computeOrder = UPDATE_PRE_TRAVERSAL;//RENDER_PRE_RENDER_PRE_TRAVERSAL;//
        if( (_computeOrder & OSGCOMPUTE_UPDATE) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( getNumChildrenRequiringUpdateTraversal() + 1 );
    }

    //------------------------------------------------------------------------------
    void Computation::checkDevice()
    {
    }


    //------------------------------------------------------------------------------
    void Computation::setParentComputation( Computation* parentComputation )
    {
        _parentComputation = parentComputation;
    }

    //------------------------------------------------------------------------------
    void Computation::addContext( osg::State& state )
    {
        if( !state.getGraphicsContext() )
        {
            osg::notify(osg::FATAL)  << "Computation::addContext() for \""
                << getName()<<"\": GLObjectsVisitor must provide a valid graphics context."
                << std::endl;

            return;
        }

        // find context
        ContextSetItr itr = _contextSet.find( state.getGraphicsContext() );
        if( itr != _contextSet.end() )
            return;

        // insert new context
        _contextSet.insert( state.getGraphicsContext() );
    }

    //------------------------------------------------------------------------------
    void Computation::addBin( osgUtil::CullVisitor& cv )
    {
        if( !cv.getState() )
        {
            osg::notify(osg::FATAL)  << "Computation::addBin() for \""
                << getName()<<"\": CullVisitor must provide a valid state."
                << std::endl;

            return;
        }

        ///////////////////////
        // SETUP REDIRECTION //
        ///////////////////////
        osgUtil::RenderBin* oldRB = cv.getCurrentRenderBin();
        if( !oldRB )
        {
            osg::notify(osg::FATAL)  
                << getName() << " [Computation::addBin()]: current CullVisitor has no active RenderBin."
                << std::endl;

            return;
        }
        const osgUtil::RenderBin::RenderBinList& rbList = oldRB->getRenderBinList();

        // We have to look for a better method to add more computation bins
        // to the same hierarchy level
        int rbNum = 0;
        if( (_computeOrder & OSGCOMPUTE_POST_RENDER) !=  OSGCOMPUTE_POST_RENDER )
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
                << getName() << " [Computation::addBin()]: cannot create ComputationBin."
                << std::endl;

            return;
        }

        pb->init( *this );

        //////////////
        // TRAVERSE //
        //////////////
        cv.setCurrentRenderBin( pb );
        cv.apply( *this );
        cv.setCurrentRenderBin( oldRB );
    }

    //------------------------------------------------------------------------------
    void Computation::launch()
    {
        if( _launchCallback ) 
        {
            (*_launchCallback)( *this ); 
        }
        else
        {
            // For all contexts launch modules
            for( ContextSetItr itr = _contextSet.begin(); itr != _contextSet.end(); ++itr )
            {
                if( !(*itr)->isCurrent() )
                    (*itr)->makeCurrent();
            
                // Activate Resource Entries
                Resource::setCurrentIdx( (*itr)->getState()->getContextID() );

                // Launch modules
                for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
                {
                    if( (*itr)->isEnabled() )
                    {
                        if( (*itr)->isClear() )
                            (*itr)->init();

                        (*itr)->launch();
                    }
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::update( osg::NodeVisitor& uv )
    {
        if( !_resourcesCollected && getParentComputation() != NULL )
        {
            // status changed from child node to topmost node
            setParentComputation( NULL );
        }

        if( _subgraphChanged )
        {
            // topmost node starts 
            // resource collection
            if( !_resourcesCollected )
                collectResources();

            // init params if not done so far 
            for( ResourceHandleListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
            {
                if( !(*itr)._resource.valid() || dynamic_cast<Module*>( (*itr)._resource.get() ) != NULL )
                    continue;

                if( (*itr)._resource->isClear() )
                    (*itr)._resource->init();
            }

            // decrement update counter when all resources have been initialized
            osg::Node::setNumChildrenRequiringUpdateTraversal( 
                osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

            // set resources changed to false
            _subgraphChanged = false;

            // if auto update is selected then we have to traverse the
            // sub-graph each update cycle
            if( getAutoCheckSubgraph() )
                subgraphChanged();
        }

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->getUpdateCallback() )
                (*(*itr)->getUpdateCallback())( *(*itr), uv );
        }
    }

    //------------------------------------------------------------------------------
    void Computation::collectResources()
    {
        if( !_resourceVisitor.valid() )
        {
            _resourceVisitor = newVisitor();
            _resourceVisitor->setComputation( this );
            if( !_resourceVisitor->init() )
            {
                osg::notify(osg::FATAL)  
                    << getName() << " [Computation::collectResources()]: cannot init resource visitor."
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
            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getEventCallback() )
                    (*(*itr)->getEventCallback())( *(*itr), ev );
            }
        }
    }  
} 