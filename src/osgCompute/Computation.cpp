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
#include <osgDB/Registry>
#include <osgUtil/CullVisitor>
#include <osgUtil/RenderBin>
#include <osgUtil/GLObjectsVisitor>
#include <osgCompute/Visitor>
#include <osgCompute/Memory>
#include <osgCompute/Computation>

namespace osgCompute
{
    class LIBRARY_EXPORT ComputationBin : public osgUtil::RenderBin 
    {
    public:
        ComputationBin(); 
        ComputationBin(osgUtil::RenderBin::SortMode mode);

        META_Object( osgCompute, ComputationBin );

        virtual bool init( Computation& computation );
        virtual void reset();

        virtual void drawImplementation( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous );
        virtual unsigned int computeNumberOfDynamicRenderLeaves() const;


        virtual bool isClear() const;
        virtual Computation* getComputation();
        virtual const Computation* getComputation() const;

        virtual void clear();

    protected:
        friend class Computation;
        virtual ~ComputationBin() { clearLocal(); }
        void clearLocal();

        virtual void launch();
        virtual void drawLeafs( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous );

        Computation*                         _computation; 
        bool                                 _clear;

    private:
        // copy constructor and operator should not be called
        ComputationBin(const ComputationBin&, const osg::CopyOp& ) {}
        ComputationBin &operator=(const ComputationBin &) { return *this; }
    };

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
        :   osg::Group()
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
            if( cv != NULL )
            {
                if( GLMemory::getContext() == NULL )
                {
                    setupContext( *cv->getState() );                    
                    // Setup state
                    if( !isDeviceReady() ) 
                    {               
                        osg::notify(osg::FATAL) 
                            << getName() << " [Computation::accept(GLObjectsVisitor)]: No valid Computation Device found."
                            << std::endl;

                        return;
                    }
                }

                if( _enabled && (_computeOrder & OSGCOMPUTE_RENDER) == OSGCOMPUTE_RENDER )
                    addBin( *cv );
                else if( (_computeOrder & OSGCOMPUTE_NORENDER) == OSGCOMPUTE_NORENDER )
					return; // Do not process the childs during rendering
				else
                    nv.apply(*this);
            }
            else if( ov != NULL )
            {
                if( GLMemory::getContext() == NULL )
                {
                    setupContext( *ov->getState() );                
                    // Setup state
                    if( !isDeviceReady() ) 
                    {               
                        osg::notify(osg::FATAL) 
                            << getName() << " [Computation::accept(GLObjectsVisitor)]: No valid Computation Device found."
                            << std::endl;

                        return;
                    }
                }

                nv.apply( *this );
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
            {

                if( _enabled && (_computeOrder & UPDATE_BEFORECHILDREN) == UPDATE_BEFORECHILDREN )
                    launch();

                if( getUpdateCallback() )
                {
                    (*getUpdateCallback())( this, &nv );
                }
                else
                {
                    applyVisitorToModules( nv );
                    nv.apply( *this );
                }

                if( _enabled && (_computeOrder & UPDATE_AFTERCHILDREN) == UPDATE_AFTERCHILDREN )
                    launch();
            }
            else if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
            {
                if( getEventCallback() )
                {
                    (*getEventCallback())( this, &nv );
                }
                else
                {
                    applyVisitorToModules( nv );
                    nv.apply( *this );
                }
            }
            else
            {
                nv.apply( *this );
            }

            nv.popFromNodePath(); 
        } 
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
    }

	//------------------------------------------------------------------------------
	const Module* Computation::getModule( const std::string& moduleIdentifier ) const
	{
		for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
			if( (*itr)->isIdentifiedBy( moduleIdentifier ) )
				return (*itr).get();

		return NULL;
	}

	//------------------------------------------------------------------------------
	Module* Computation::getModule( const std::string& moduleIdentifier )
	{
		for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
			if( (*itr)->isIdentifiedBy( moduleIdentifier ) )
				return (*itr).get();

		return NULL;
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
        for( ResourceHandleListCnstItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            if(	(*itr)._resource.valid() && (*itr)._resource->isIdentifiedBy(handle)  )
                return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Computation::addResource( Resource& resource, bool serialize )
    {
        if( hasResource(resource) )
            return;

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            (*itr)->acceptResource( resource );

		ResourceHandle newHandle;
		newHandle._resource = &resource;
		newHandle._serialize = serialize;
		_resources.push_back( newHandle );

		if( resource.isClear() ) resource.init();
    }

    //------------------------------------------------------------------------------
    void Computation::exchangeResource( Resource& newResource, bool serialize /*= true */ )
    {
        IdentifierSet& ids = newResource.getIdentifiers();
        for( ResourceHandleListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            bool exchange = false;
            for( IdentifierSetItr idItr = ids.begin(); idItr != ids.end(); ++idItr )
            {
                if( (*itr)._resource->isIdentifiedBy( (*idItr) ) )
                {
                    exchange = true;
                }
            }

            if( exchange )
            {
                // Remove and add resource
                for( ModuleListItr modItr = _modules.begin(); modItr != _modules.end(); ++modItr )
                {
                    (*modItr)->removeResource( *((*itr)._resource) );
                    (*modItr)->acceptResource( newResource );
                }
                // Remove resource from list
                itr = _resources.erase( itr );
            }
        }

        // Add new resource
        ResourceHandle newHandle;
        newHandle._resource = &newResource;
        newHandle._serialize = serialize;
        _resources.push_back( newHandle );

		if( newResource.isClear() ) newResource.init();
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
	bool Computation::isResourceSerialized( Resource& resource ) const
	{
		for( ResourceHandleListCnstItr itr = _resources.begin();
			itr != _resources.end();
			++itr )
		{
			if( (*itr)._resource == &resource )
				return (*itr)._serialize;
		}

		return false;
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
        if( state != NULL && GLMemory::getContext() == state->getGraphicsContext() )
        {
            // Make context the current context
            if( !GLMemory::getContext()->isCurrent() && GLMemory::getContext()->isRealized() )
                GLMemory::getContext()->makeCurrent();

            // Release all resources associated with the current context
            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
                (*itr)->releaseObjects();

            for( ResourceHandleListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
                (*itr)._resource->releaseObjects();
        }

        Group::releaseGLObjects( state );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------   
    void Computation::clearLocal()
    {
        removeResources();
        removeModules();
        _launchCallback = NULL;
        _enabled = true;

        // clear node or group related members
        removeChildren(0,osg::Group::getNumChildren());
        setDataVariance( osg::Object::DYNAMIC );
        setUpdateCallback( NULL );
        setEventCallback( NULL );

        // setup computation order
        _computeOrder = UPDATE_BEFORECHILDREN;
        if( (_computeOrder & OSGCOMPUTE_UPDATE) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( getNumChildrenRequiringUpdateTraversal() + 1 );
    }

    //------------------------------------------------------------------------------
    void Computation::checkDevice()
    {
    }

    //------------------------------------------------------------------------------
    void Computation::setupContext( osg::State& state )
    {
        if( !state.getGraphicsContext() )
        {
            osg::notify(osg::WARN)  << "Computation::setupContext() for \""
                << getName()<<"\": GLObjectsVisitor must provide a valid graphics context."
                << std::endl;

            return;
        }

        if( NULL != GLMemory::getContext() && 
            GLMemory::getContext()->getState()->getContextID() != state.getContextID() )
        {
            osg::notify(osg::WARN)  << "Computation::setupContext() for \""
                << getName()<<"\": GLObjectsVisitor can handle only a single context."
                << " However multiple contexts are detected."
                << " Please make shure to share a computation context by several windows."
                << std::endl;

            return;
        }

        if( GLMemory::getContext() == NULL )
            GLMemory::bindToContext( *state.getGraphicsContext() );
    }

    

    //------------------------------------------------------------------------------
    void Computation::addBin( osgUtil::CullVisitor& cv )
    {
        if( !cv.getState() )
        {
            osg::notify(osg::WARN)  << "Computation::addBin() for \""
                << getName()<<"\": CullVisitor must provide a valid state."
                << std::endl;

            return;
        }

        if( GLMemory::getContext() == NULL )
            return;

		if( cv.getState()->getContextID() != GLMemory::getContext()->getState()->getContextID() )
			return;

        ///////////////////////
        // SETUP REDIRECTION //
        ///////////////////////
        osgUtil::RenderBin* oldRB = cv.getCurrentRenderBin();
        if( !oldRB )
        {
            osg::notify(osg::WARN)  
                << getName() << " [Computation::addBin()]: current CullVisitor has no active RenderBin."
                << std::endl;

            return;
        }
        const osgUtil::RenderBin::RenderBinList& rbList = oldRB->getRenderBinList();

        // We have to look for a better method to add more computation bins
        // to the same hierarchy level
        int rbNum = 0;
        if( (_computeOrder & OSGCOMPUTE_POSTRENDER) !=  OSGCOMPUTE_POSTRENDER )
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
        // Check if graphics context exist
        // or return otherwise
        if( NULL != GLMemory::getContext() && GLMemory::getContext()->isRealized() )
        {       
                    
            // Make context the current context
            // Annotation: better avoid this! It seems to cause problems with several windows (widgets)
            // which share GL context! And: buffers which are also used in GL do not need to be "mapped" to host
            // first when using shared contexts.
            //if( !GLMemory::getContext()->isCurrent() ) 
            //    GLMemory::getContext()->makeCurrent();

            // Launch modules
            if( _launchCallback.valid() ) 
            {
                (*_launchCallback)( *this ); 
            }
            else
            {
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
    void Computation::applyVisitorToModules( osg::NodeVisitor& nv )
    {
        if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
        {
            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getEventCallback() )
                    (*(*itr)->getEventCallback())( *(*itr), nv );
            }
        }
        else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
        {
            for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            {
                if( (*itr)->getUpdateCallback() )
                    (*(*itr)->getUpdateCallback())( *(*itr), nv );
            }
        }
    }  

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// BIN IMPLEMENTATION ///////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	class RegisterBinProxy
	{
	public:
		RegisterBinProxy( const std::string& binName, osgUtil::RenderBin* proto )
		{
			_rb = proto;
			osgUtil::RenderBin::addRenderBinPrototype( binName, _rb.get() );
		}
	protected:
		osg::ref_ptr<osgUtil::RenderBin> _rb;
	};

	RegisterBinProxy registerComputationBinProxy("osgCompute::ComputationBin", new osgCompute::ComputationBin );

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	ComputationBin::ComputationBin()
		: osgUtil::RenderBin()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	ComputationBin::ComputationBin( osgUtil::RenderBin::SortMode mode )
		: osgUtil::RenderBin( mode )
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void ComputationBin::clear()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void ComputationBin::drawImplementation( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
	{ 
		osg::State& state = *renderInfo.getState();

		unsigned int numToPop = (previous ? osgUtil::StateGraph::numToPop(previous->_parent) : 0);
		if (numToPop>1) --numToPop;
		unsigned int insertStateSetPosition = state.getStateSetStackSize() - numToPop;

		if (_stateset.valid())
		{
			state.insertStateSet(insertStateSetPosition, _stateset.get());
		}

		///////////////////
		// DRAW PRE BINS //
		///////////////////
		osgUtil::RenderBin::RenderBinList::iterator rbitr;
		for(rbitr = _bins.begin();
			rbitr!=_bins.end() && rbitr->first<0;
			++rbitr)
		{
			rbitr->second->draw(renderInfo,previous);
		}


		if( (_computation->getComputeOrder() & OSGCOMPUTE_BEFORECHILDREN ) == OSGCOMPUTE_BEFORECHILDREN )
		{
			if( _computation->getLaunchCallback() ) 
				(*_computation->getLaunchCallback())( *_computation ); 
			else launch(); 

			// don't forget to decrement dynamic object count
			renderInfo.getState()->decrementDynamicObjectCount();
		}

		// render sub-graph leafs
		if( (_computation->getComputeOrder() & OSGCOMPUTE_NOCHILDREN ) != OSGCOMPUTE_NOCHILDREN )
			drawLeafs(renderInfo, previous );

		if( (_computation->getComputeOrder() & OSGCOMPUTE_BEFORECHILDREN ) != OSGCOMPUTE_BEFORECHILDREN )
		{
			if( _computation->getLaunchCallback() ) 
				(*_computation->getLaunchCallback())( *_computation ); 
			else launch();  

			// don't forget to decrement dynamic object count
			renderInfo.getState()->decrementDynamicObjectCount();
		}

		////////////////////
		// DRAW POST BINS //
		////////////////////
		for(;
			rbitr!=_bins.end();
			++rbitr)
		{
			rbitr->second->draw(renderInfo,previous);
		}


		if (_stateset.valid())
		{
			state.removeStateSet(insertStateSetPosition);
		}
	}

	//------------------------------------------------------------------------------
	unsigned int ComputationBin::computeNumberOfDynamicRenderLeaves() const
	{
		// increment dynamic object count to execute modules
		return osgUtil::RenderBin::computeNumberOfDynamicRenderLeaves() + 1;
	}

	//------------------------------------------------------------------------------
	bool ComputationBin::init( Computation& computation )
	{
		// COMPUTATION 
		_computation = &computation;

		// OBJECT 
		setName( _computation->getName() );
		setDataVariance( _computation->getDataVariance() );

		_clear = false;
		return true;
	}

	//------------------------------------------------------------------------------
	void ComputationBin::reset()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool ComputationBin::isClear() const
	{ 
		return _clear; 
	}

	//------------------------------------------------------------------------------
	Computation* ComputationBin::getComputation()
	{ 
		return _computation; 
	}

	//------------------------------------------------------------------------------
	const Computation* ComputationBin::getComputation() const
	{ 
		return _computation; 
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void ComputationBin::clearLocal()
	{
		_computation = NULL;
		_clear = true;

		osgUtil::RenderBin::reset();
	}

	//------------------------------------------------------------------------------
	void ComputationBin::launch()
	{
		if( _clear || !_computation  )
			return;

		// Launch modules
		ModuleList& modules = _computation->getModules();
		for( ModuleListCnstItr itr = modules.begin(); itr != modules.end(); ++itr )
		{
			if( (*itr)->isEnabled() )
			{
				if( (*itr)->isClear() )
					(*itr)->init();

				(*itr)->launch();
			}
		}
	}

	//------------------------------------------------------------------------------
	void ComputationBin::drawLeafs( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
	{
		// draw fine grained ordering.
		for(osgUtil::RenderBin::RenderLeafList::iterator rlitr= _renderLeafList.begin();
			rlitr!= _renderLeafList.end();
			++rlitr)
		{
			osgUtil::RenderLeaf* rl = *rlitr;
			rl->render(renderInfo,previous);
			previous = rl;
		}

		// draw coarse grained ordering.
		for(osgUtil::RenderBin::StateGraphList::iterator oitr=_stateGraphList.begin();
			oitr!=_stateGraphList.end();
			++oitr)
		{

			for(osgUtil::StateGraph::LeafList::iterator dw_itr = (*oitr)->_leaves.begin();
				dw_itr != (*oitr)->_leaves.end();
				++dw_itr)
			{
				osgUtil::RenderLeaf* rl = dw_itr->get();
				rl->render(renderInfo,previous);
				previous = rl;

			}
		}
	}
} 