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
				getOrCreateContext( *cv->getState() );

				if( (_computeOrder & OSGCOMPUTE_RENDER) == OSGCOMPUTE_RENDER )
					addBin( *cv );
				else
					nv.apply(*this);
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

					if( (_computeOrder & UPDATE_PRE_TRAVERSAL) == UPDATE_PRE_TRAVERSAL )
						launch();

                }
				else if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
				{
					handleevent( nv );
				}

                nv.apply( *this );

				if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR && 
					(_computeOrder & UPDATE_POST_TRAVERSAL) == UPDATE_POST_TRAVERSAL )
					launch();
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
    void Computation::removeModule( const std::string& moduleHandle )
    {
        ModuleListItr itr = _modules.begin();
        while( itr != _modules.end() )
        {
            if( (*itr)->isAddressedByHandle( moduleHandle ) )
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

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->acceptResource( resource );

        _resources.insert( std::make_pair< Resource*, osg::ref_ptr<Resource> > ( &resource, &resource ) );
        subgraphChanged();
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
        ResourceMapItr itr = _resources.find( &resource );
        if( itr != _resources.end() )
        {
            for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                (*moditr)->removeResource( resource );

             _resources.erase( itr );
             subgraphChanged();
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
                for( ModuleListItr moditr = _modules.begin(); moditr != _modules.end(); ++moditr )
                    (*moditr)->removeResource( *curResource );
            }

            _resources.erase( itr );
            itr = _resources.begin();
        }
        subgraphChanged();
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
	Computation::ComputeOrder Computation::getComputeOrder()
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
		for( ContextMapItr itr = _contextMap.begin();
			 itr != _contextMap.end();
			 ++itr )
		{
			if( (*itr).second->getGraphicsContext()->getState() == state )
			{
				(*itr).second->clearResources();
				_contextMap.erase( itr );
				break;
			}
		}

		Group::releaseGLObjects( state );
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
        _contextMap.clear();

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
		if( !context.isConnectedWithGraphicsContext() ||
			context.getGraphicsContext()->getState() == NULL )
			return false;

		if( _parentComputation == NULL )
		{
			_contextMap[ context.getGraphicsContext()->getState()->getContextID() ] = &context;
		
			// Pass on context to subgraph
			distributeContext( context );
		}
		else
		{
			// Search for topmost computation
			Computation* topComp = this;
			while( NULL != topComp->getParentComputation() )
				topComp = topComp->getParentComputation();

			topComp->setContext( context );
		}

        return true;
    }

    //------------------------------------------------------------------------------
    Context* Computation::getContext( osg::State& state )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        ContextMapItr itr = _contextMap.find( state.getContextID() );
        if( itr == _contextMap.end() )
            return NULL;

        return (*itr).second.get();
    }

    //------------------------------------------------------------------------------
	Context* Computation::getOrCreateContext( osg::State& state )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

		// find or create context
		bool contextCreated = false;
        Context* context = NULL;
        ContextMapItr itr = _contextMap.find( state.getContextID() );
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

			context->connectWithGraphicsContext( *state.getGraphicsContext() );
            _contextMap.insert( std::make_pair< unsigned int, osg::ref_ptr<Context> >( state.getContextID(), context) );
			context->init();
			contextCreated = true;
        }
        else
        {
            context = (*itr).second.get();
        }

		// traverse subgraph and pass on context 
		if( contextCreated || getAutoCheckSubgraph() )
			distributeContext( *context );

        return context;
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

		Context* ctx = getContext( *cv.getState() );
		if( !ctx )
		{
			osg::notify(osg::FATAL)  
				<< "Computation::addBin() for \""<<getName()<<"\": cannot find valid context."
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
                << "Computation::addBin() for \""<<getName()<<"\": current CullVisitor has no active RenderBin."
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
                << "Computation::addBin() for \""<<getName()<<"\": cannot create ComputationBin."
                << std::endl;

            return;
        }

        pb->init( *this );
        pb->setContext( *ctx );

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
		// For all contexts launch modules
		for( ContextMapItr itr = _contextMap.begin(); itr != _contextMap.end(); ++itr )
		{
			Context* curCtx = itr->second.get();
			if( curCtx->isClear() )
				continue;

			// Apply context 
			curCtx->apply();

			// Launch modules
			for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
			{
				if( (*itr)->isEnabled() )
					(*itr)->launch( *curCtx );
			}
		}
	}

	//------------------------------------------------------------------------------
	void Computation::acceptContext( Context& context )
	{
		if( !context.isConnectedWithGraphicsContext() ||
			context.getGraphicsContext()->getState() == NULL )
			return;

		_contextMap[ context.getGraphicsContext()->getState()->getContextID() ] = &context;
	}

	//------------------------------------------------------------------------------
	void osgCompute::Computation::removeContext( osg::State& state )
	{
		ContextMapItr itr = _contextMap.find( state.getContextID() );
		if( itr != _contextMap.end() )
			_contextMap.erase(itr);
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

        if( _subgraphChanged )
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
                if( !curResource || dynamic_cast<Module*>( curResource ) != NULL )
                    continue;

                if( curResource->isClear() )
                    curResource->init();
            }

            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->isClear() )
                    (*itr)->init();
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
			for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
			{
				if( (*itr)->getEventCallback() )
					(*(*itr)->getEventCallback())( *(*itr), ev );
			}
		}
	}  
}
