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
#include <osgCompute/Program>

namespace osgCompute
{
    class LIBRARY_EXPORT ProgramBin : public osgUtil::RenderBin 
    {
    public:
        ProgramBin(); 
        ProgramBin(osgUtil::RenderBin::SortMode mode);

        META_Object( osgCompute, ProgramBin );

        virtual bool init( Program& program );
        virtual void reset();

        virtual void drawImplementation( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous );
        virtual unsigned int computeNumberOfDynamicRenderLeaves() const;


        virtual bool isClear() const;
        virtual Program* getProgram();
        virtual const Program* getProgram() const;

        virtual void clear();

    protected:
        friend class Program;
        virtual ~ProgramBin() { clearLocal(); }
        void clearLocal();

        virtual void launch();
        virtual void drawLeafs( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous );

        Program* _program; 
        bool     _clear;

    private:
        // copy constructor and operator should not be called
        ProgramBin(const ProgramBin&, const osg::CopyOp& ) {}
        ProgramBin &operator=(const ProgramBin &) { return *this; }
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    bool s_deviceReady = false;

    //------------------------------------------------------------------------------
    bool isDeviceReady()
    {
        return s_deviceReady;
    }

    //------------------------------------------------------------------------------
    void setDeviceReady()
    {
        s_deviceReady = true;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------ 
    Program::Program() 
        :   osg::Group()
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------   
    void Program::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Program::accept(osg::NodeVisitor& nv) 
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
                            << getName() << " [Program::accept(GLObjectsVisitor)]: No valid Program Device found."
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
                            << getName() << " [Program::accept(GLObjectsVisitor)]: No valid Program Device found."
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
                    applyVisitorToComputations( nv );
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
                    applyVisitorToComputations( nv );
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
    void Program::addComputation( Computation& computation )
    {
        Resource* curResource = NULL;
        for( ResourceHandleListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            curResource = (*itr)._resource.get();
            if( !curResource )
                continue;

            computation.acceptResource( *curResource );
        }

        // increment traversal counter if required
        if( computation.getEventCallback() )
            osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() + 1 );

        if( computation.getUpdateCallback() )
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );


        _computations.push_back( &computation );
    }

    //------------------------------------------------------------------------------
    void Program::removeComputation( Computation& computation )
    {
        for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
        {
            if( (*itr) == &computation )
            {
                // decrement traversal counter if necessary
                if( computation.getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( computation.getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                _computations.erase( itr );
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Program::removeComputation( const std::string& computationIdentifier )
    {
        ComputationListItr itr = _computations.begin();
        while( itr != _computations.end() )
        {
            if( (*itr)->isIdentifiedBy( computationIdentifier ) )
            {
                // decrement traversal counter if necessary
                if( (*itr)->getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( (*itr)->getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );


                _computations.erase( itr );
                itr = _computations.begin();
            }
            else
            {
                ++itr;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Program::removeComputations()
    {
        ComputationListItr itr;
        while( !_computations.empty() )
        {
            itr = _computations.begin();

            Computation* curComputation = (*itr).get();
            if( curComputation != NULL )
            {
                // decrement traversal counter if necessary
                if( curComputation->getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( curComputation->getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );
            }
            _computations.erase( itr );
        }
    }

	//------------------------------------------------------------------------------
	const Computation* Program::getComputation( const std::string& computationIdentifier ) const
	{
		for( ComputationListCnstItr itr = _computations.begin(); itr != _computations.end(); ++itr )
			if( (*itr)->isIdentifiedBy( computationIdentifier ) )
				return (*itr).get();

		return NULL;
	}

	//------------------------------------------------------------------------------
	Computation* Program::getComputation( const std::string& computationIdentifier )
	{
		for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
			if( (*itr)->isIdentifiedBy( computationIdentifier ) )
				return (*itr).get();

		return NULL;
	}


    //------------------------------------------------------------------------------
    bool Program::hasComputation( const std::string& computationIdentifier ) const
    {
        for( ComputationListCnstItr itr = _computations.begin(); itr != _computations.end(); ++itr )
            if( (*itr)->isIdentifiedBy( computationIdentifier ) )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Program::hasComputation( Computation& computation ) const
    {
        for( ComputationListCnstItr itr = _computations.begin(); itr != _computations.end(); ++itr )
            if( (*itr) == &computation )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Program::hasComputations() const 
    { 
        return !_computations.empty(); 
    }

    //------------------------------------------------------------------------------
    ComputationList& Program::getComputations() 
    { 
        return _computations; 
    }

    //------------------------------------------------------------------------------
    const ComputationList& Program::getComputations() const 
    { 
        return _computations; 
    }

    //------------------------------------------------------------------------------
    unsigned int Program::getNumComputations() const 
    { 
        return _computations.size(); 
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Program::hasResource( Resource& resource ) const
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
    bool osgCompute::Program::hasResource( const std::string& handle ) const
    {
        for( ResourceHandleListCnstItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            if(	(*itr)._resource.valid() && (*itr)._resource->isIdentifiedBy(handle)  )
                return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Program::addResource( Resource& resource, bool serialize )
    {
        if( hasResource(resource) )
            return;

        for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
            (*itr)->acceptResource( resource );

		ResourceHandle newHandle;
		newHandle._resource = &resource;
		newHandle._serialize = serialize;
		_resources.push_back( newHandle );

		if( resource.isClear() ) resource.init();
    }

    //------------------------------------------------------------------------------
    void Program::exchangeResource( Resource& newResource, bool serialize /*= true */ )
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
                for( ComputationListItr modItr = _computations.begin(); modItr != _computations.end(); ++modItr )
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
    void osgCompute::Program::removeResource( const std::string& handle )
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
                for( ComputationListItr moditr = _computations.begin(); moditr != _computations.end(); ++moditr )
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
    void Program::removeResource( Resource& resource )
    {
		for( ResourceHandleListItr itr = _resources.begin();
			itr != _resources.end();
			++itr )
		{
			if( (*itr)._resource == &resource )
			{
				for( ComputationListItr moditr = _computations.begin(); moditr != _computations.end(); ++moditr )
					(*moditr)->removeResource( resource );

				_resources.erase( itr );
				return;
			}
		}

    }

    //------------------------------------------------------------------------------
    void Program::removeResources()
    {
        ResourceHandleListItr itr = _resources.begin();
        while( itr != _resources.end() )
        {
            Resource* curResource = (*itr)._resource.get();
            if( curResource != NULL )
            {
                for( ComputationListItr moditr = _computations.begin(); moditr != _computations.end(); ++moditr )
                    (*moditr)->removeResource( *curResource );
            }

            _resources.erase( itr );
            itr = _resources.begin();
        }
    }

    //------------------------------------------------------------------------------
    ResourceHandleList& Program::getResources()
    {
        return _resources;
    }

    //------------------------------------------------------------------------------
    const ResourceHandleList& Program::getResources() const
    {
        return _resources;
    }

	//------------------------------------------------------------------------------
	bool Program::isResourceSerialized( Resource& resource ) const
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
    void Program::setLaunchCallback( LaunchCallback* lc ) 
    { 
        if( lc == _launchCallback )
            return;

        _launchCallback = lc; 
    }

    //------------------------------------------------------------------------------
    LaunchCallback* Program::getLaunchCallback() 
    { 
        return _launchCallback; 
    }

    //------------------------------------------------------------------------------
    const LaunchCallback* Program::getLaunchCallback() const 
    { 
        return _launchCallback; 
    }

    //------------------------------------------------------------------------------
    void Program::setComputeOrder( Program::ComputeOrder co )
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
    Program::ComputeOrder Program::getComputeOrder() const
    {
        return _computeOrder;
    }

    //------------------------------------------------------------------------------
    void Program::enable() 
    { 
        _enabled = true; 
    }

    //------------------------------------------------------------------------------
    void Program::disable() 
    { 
        _enabled = false; 
    }

    //------------------------------------------------------------------------------
    bool Program::isEnabled() const
    {
        return _enabled;
    }

    //------------------------------------------------------------------------------
    void Program::releaseGLObjects( osg::State* state ) const
    {
        if( state != NULL && GLMemory::getContext() == state->getGraphicsContext() )
        {
            // Make context the current context
            if( !GLMemory::getContext()->isCurrent() && GLMemory::getContext()->isRealized() )
                GLMemory::getContext()->makeCurrent();

            // Release all resources associated with the current context
            for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
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
    void Program::clearLocal()
    {
        removeResources();
        removeComputations();
        _launchCallback = NULL;
        _enabled = true;

        // clear node or group related members
        removeChildren(0,osg::Group::getNumChildren());
        setDataVariance( osg::Object::DYNAMIC );
        setUpdateCallback( NULL );
        setEventCallback( NULL );

        // setup program order
        _computeOrder = UPDATE_BEFORECHILDREN;
        if( (_computeOrder & OSGCOMPUTE_UPDATE) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( getNumChildrenRequiringUpdateTraversal() + 1 );
    }

    //------------------------------------------------------------------------------
    void Program::checkDevice()
    {
    }

    //------------------------------------------------------------------------------
    void Program::setupContext( osg::State& state )
    {
        if( !state.getGraphicsContext() )
        {
            osg::notify(osg::WARN)  << "Program::setupContext() for \""
                << getName()<<"\": GLObjectsVisitor must provide a valid graphics context."
                << std::endl;

            return;
        }

        if( NULL != GLMemory::getContext() && 
            GLMemory::getContext()->getState()->getContextID() != state.getContextID() )
        {
            osg::notify(osg::WARN)  << "Program::setupContext() for \""
                << getName()<<"\": GLObjectsVisitor can handle only a single context."
                << " However multiple contexts are detected."
                << " Please make shure to share a program context by several windows."
                << std::endl;

            return;
        }

        if( GLMemory::getContext() == NULL )
            GLMemory::bindToContext( *state.getGraphicsContext() );
    }

    

    //------------------------------------------------------------------------------
    void Program::addBin( osgUtil::CullVisitor& cv )
    {
        if( !cv.getState() )
        {
            osg::notify(osg::WARN)  << "Program::addBin() for \""
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
                << getName() << " [Program::addBin()]: current CullVisitor has no active RenderBin."
                << std::endl;

            return;
        }
        const osgUtil::RenderBin::RenderBinList& rbList = oldRB->getRenderBinList();

        // We have to look for a better method to add more program bins
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

        ProgramBin* pb = 
            dynamic_cast<ProgramBin*>( oldRB->find_or_insert(rbNum,compBinName) );

        if( !pb )
        {
            osg::notify(osg::FATAL)  
                << getName() << " [Program::addBin()]: cannot create ProgramBin."
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
    void Program::launch()
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

            // Launch computations
            if( _launchCallback.valid() ) 
            {
                (*_launchCallback)( *this ); 
            }
            else
            {
                for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
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
    void Program::applyVisitorToComputations( osg::NodeVisitor& nv )
    {
        if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
        {
            for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
            {
                if( (*itr)->getEventCallback() )
                    (*(*itr)->getEventCallback())( *(*itr), nv );
            }
        }
        else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
        {
            for( ComputationListItr itr = _computations.begin(); itr != _computations.end(); ++itr )
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

	RegisterBinProxy registerProgramBinProxy("osgCompute::ProgramBin", new osgCompute::ProgramBin );

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	ProgramBin::ProgramBin()
		: osgUtil::RenderBin()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	ProgramBin::ProgramBin( osgUtil::RenderBin::SortMode mode )
		: osgUtil::RenderBin( mode )
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void ProgramBin::clear()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void ProgramBin::drawImplementation( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
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


		if( (_program->getComputeOrder() & OSGCOMPUTE_BEFORECHILDREN ) == OSGCOMPUTE_BEFORECHILDREN )
		{
			if( _program->getLaunchCallback() ) 
				(*_program->getLaunchCallback())( *_program ); 
			else launch(); 

			// don't forget to decrement dynamic object count
			renderInfo.getState()->decrementDynamicObjectCount();
		}

		// render sub-graph leafs
		if( (_program->getComputeOrder() & OSGCOMPUTE_NOCHILDREN ) != OSGCOMPUTE_NOCHILDREN )
			drawLeafs(renderInfo, previous );

		if( (_program->getComputeOrder() & OSGCOMPUTE_BEFORECHILDREN ) != OSGCOMPUTE_BEFORECHILDREN )
		{
			if( _program->getLaunchCallback() ) 
				(*_program->getLaunchCallback())( *_program ); 
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
	unsigned int ProgramBin::computeNumberOfDynamicRenderLeaves() const
	{
		// increment dynamic object count to execute computations
		return osgUtil::RenderBin::computeNumberOfDynamicRenderLeaves() + 1;
	}

	//------------------------------------------------------------------------------
	bool ProgramBin::init( Program& program )
	{
		// COMPUTATION 
		_program = &program;

		// OBJECT 
		setName( _program->getName() );
		setDataVariance( _program->getDataVariance() );

		_clear = false;
		return true;
	}

	//------------------------------------------------------------------------------
	void ProgramBin::reset()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool ProgramBin::isClear() const
	{ 
		return _clear; 
	}

	//------------------------------------------------------------------------------
	Program* ProgramBin::getProgram()
	{ 
		return _program; 
	}

	//------------------------------------------------------------------------------
	const Program* ProgramBin::getProgram() const
	{ 
		return _program; 
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void ProgramBin::clearLocal()
	{
		_program = NULL;
		_clear = true;

		osgUtil::RenderBin::reset();
	}

	//------------------------------------------------------------------------------
	void ProgramBin::launch()
	{
		if( _clear || !_program  )
			return;

		// Launch computations
		ComputationList& computations = _program->getComputations();
		for( ComputationListCnstItr itr = computations.begin(); itr != computations.end(); ++itr )
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
	void ProgramBin::drawLeafs( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
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