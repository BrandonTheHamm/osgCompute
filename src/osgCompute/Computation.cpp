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
//#include <osgUtil/RenderBin>
#include <osgUtil/RenderStage>
#include <osgUtil/GLObjectsVisitor>
#include <osgCompute/Visitor>
#include <osgCompute/Memory>
#include <osgCompute/Computation>

namespace osgCompute
{
    class LIBRARY_EXPORT ComputationBin : public osgUtil::RenderStage 
    {
    public:
        ComputationBin(); 
        ComputationBin(osgUtil::RenderBin::SortMode mode);

        META_Object( osgCompute, ComputationBin );

        virtual bool setup( Computation& computation );
        virtual void reset();

        virtual void draw( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous );
        virtual void drawImplementation(osg::RenderInfo& renderInfo,osgUtil::RenderLeaf*& previous);
        virtual void drawInner(osg::RenderInfo& renderInfo,osgUtil::RenderLeaf*& previous, bool& doCopyTexture);

        virtual unsigned int computeNumberOfDynamicRenderLeaves() const;

        virtual Computation* getComputation();
        virtual const Computation* getComputation() const;

    protected:
        friend class Computation;
        virtual ~ComputationBin() { clearLocal(); }
        void clearLocal();

        virtual void launch();
        virtual void drawLeafs( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous );

        Computation* _computation; 

    private:
        // copy constructor and operator should not be called
        ComputationBin(const ComputationBin&, const osg::CopyOp& ) {}
        ComputationBin &operator=(const ComputationBin &) { return *this; }
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ComputationBin::ComputationBin()
        : osgUtil::RenderStage()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    ComputationBin::ComputationBin( osgUtil::RenderBin::SortMode mode )
        : osgUtil::RenderStage( mode )
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void ComputationBin::drawInner( osg::RenderInfo& renderInfo,osgUtil::RenderLeaf*& previous, bool& doCopyTexture )
    {
        osgUtil::RenderBin::draw(renderInfo,previous);
    }

    //------------------------------------------------------------------------------
    void ComputationBin::draw( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
    { 
        if (_stageDrawnThisFrame) return;
        _stageDrawnThisFrame = true;

        // Render all the pre draw callbacks
        drawPreRenderStages(renderInfo,previous);

        bool doCopyTexture = false;
        // Draw bins, renderleafs and launch kernels
        drawInner(renderInfo,previous,doCopyTexture);

        // Render all the post draw callbacks
        drawPostRenderStages(renderInfo,previous);
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

        if( _computation->getLaunchCallback() ) 
            (*_computation->getLaunchCallback())( *_computation ); 
        else launch();  

        // don't forget to decrement dynamic object count
        renderInfo.getState()->decrementDynamicObjectCount();

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
        // increment dynamic object count to execute programs
        return osgUtil::RenderBin::computeNumberOfDynamicRenderLeaves() + 1;
    }

    //------------------------------------------------------------------------------
    bool ComputationBin::setup( Computation& computation )
    {
        // COMPUTATION 
        _computation = &computation;

        // OBJECT 
        setName( _computation->getName() );
        setDataVariance( _computation->getDataVariance() );

        return true;
    }

    //------------------------------------------------------------------------------
    void ComputationBin::reset()
    {
        clearLocal();
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
        _stageDrawnThisFrame = false;
        _computation = NULL;
        osgUtil::RenderStage::reset();
    }

    //------------------------------------------------------------------------------
    void ComputationBin::launch()
    {
        if( !_computation  )
            return;

        // Launch programs
        ProgramList& programs = _computation->getPrograms();
        for( ProgramListCnstItr itr = programs.begin(); itr != programs.end(); ++itr )
        {
            if( (*itr)->isEnabled() )
            {
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

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------ 
    Computation::Computation() 
        :   osg::Group()
    { 
        _launchCallback = NULL;
        _enabled = true;

        // setup computation order
        _computeOrder = UPDATE_BEFORECHILDREN;
        if( (_computeOrder & OSGCOMPUTE_UPDATE) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( 1 );
    }

    //------------------------------------------------------------------------------
    void Computation::accept(osg::NodeVisitor& nv) 
    { 
        if( nv.validNodeMask(*this) ) 
        {  
            nv.pushOntoNodePath(this);

            osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>( &nv );
            if( cv != NULL )
            {
                if( _enabled && (_computeOrder & OSGCOMPUTE_RENDER) == OSGCOMPUTE_RENDER )
                    addBin( *cv );
                else if( (_computeOrder & OSGCOMPUTE_NORENDER) == OSGCOMPUTE_NORENDER )
					return; // Do not process the childs during rendering
				else
                    nv.apply(*this);
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
                    applyVisitorToPrograms( nv );
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
                    applyVisitorToPrograms( nv );
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
    void Computation::addProgram( Program& program )
    {
        Resource* curResource = NULL;
        for( ResourceHandleListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            curResource = (*itr)._resource.get();
            if( !curResource )
                continue;

            program.acceptResource( *curResource );
        }

        // increment traversal counter if required
        if( program.getEventCallback() )
            osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() + 1 );

        if( program.getUpdateCallback() )
            osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() + 1 );


        _programs.push_back( &program );
    }

    //------------------------------------------------------------------------------
    void Computation::removeProgram( Program& program )
    {
        for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
        {
            if( (*itr) == &program )
            {
                // decrement traversal counter if necessary
                if( program.getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( program.getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );

                _programs.erase( itr );
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removeProgram( const std::string& programIdentifier )
    {
        ProgramListItr itr = _programs.begin();
        while( itr != _programs.end() )
        {
            if( (*itr)->isIdentifiedBy( programIdentifier ) )
            {
                // decrement traversal counter if necessary
                if( (*itr)->getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( (*itr)->getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );


                _programs.erase( itr );
                itr = _programs.begin();
            }
            else
            {
                ++itr;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::removePrograms()
    {
        ProgramListItr itr;
        while( !_programs.empty() )
        {
            itr = _programs.begin();

            Program* curProgram = (*itr).get();
            if( curProgram != NULL )
            {
                // decrement traversal counter if necessary
                if( curProgram->getEventCallback() )
                    osg::Node::setNumChildrenRequiringEventTraversal( osg::Node::getNumChildrenRequiringEventTraversal() - 1 );

                if( curProgram->getUpdateCallback() )
                    osg::Node::setNumChildrenRequiringUpdateTraversal( osg::Node::getNumChildrenRequiringUpdateTraversal() - 1 );
            }
            _programs.erase( itr );
        }
    }

	//------------------------------------------------------------------------------
	const Program* Computation::getProgram( const std::string& programIdentifier ) const
	{
		for( ProgramListCnstItr itr = _programs.begin(); itr != _programs.end(); ++itr )
			if( (*itr)->isIdentifiedBy( programIdentifier ) )
				return (*itr).get();

		return NULL;
	}

	//------------------------------------------------------------------------------
	Program* Computation::getProgram( const std::string& programIdentifier )
	{
		for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
			if( (*itr)->isIdentifiedBy( programIdentifier ) )
				return (*itr).get();

		return NULL;
	}


    //------------------------------------------------------------------------------
    bool Computation::hasProgram( const std::string& programIdentifier ) const
    {
        for( ProgramListCnstItr itr = _programs.begin(); itr != _programs.end(); ++itr )
            if( (*itr)->isIdentifiedBy( programIdentifier ) )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Computation::hasProgram( Program& program ) const
    {
        for( ProgramListCnstItr itr = _programs.begin(); itr != _programs.end(); ++itr )
            if( (*itr) == &program )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool Computation::hasPrograms() const 
    { 
        return !_programs.empty(); 
    }

    //------------------------------------------------------------------------------
    ProgramList& Computation::getPrograms() 
    { 
        return _programs; 
    }

    //------------------------------------------------------------------------------
    const ProgramList& Computation::getPrograms() const 
    { 
        return _programs; 
    }

    //------------------------------------------------------------------------------
    unsigned int Computation::getNumPrograms() const 
    { 
        return _programs.size(); 
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

        for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
            (*itr)->acceptResource( resource );

		ResourceHandle newHandle;
		newHandle._resource = &resource;
		newHandle._serialize = serialize;
		_resources.push_back( newHandle );
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
                for( ProgramListItr modItr = _programs.begin(); modItr != _programs.end(); ++modItr )
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
                for( ProgramListItr moditr = _programs.begin(); moditr != _programs.end(); ++moditr )
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
				for( ProgramListItr moditr = _programs.begin(); moditr != _programs.end(); ++moditr )
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
                for( ProgramListItr moditr = _programs.begin(); moditr != _programs.end(); ++moditr )
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
    void Computation::setComputeOrder( Computation::ComputeOrder co, int orderNum/* = 0 */)
    {
        // deactivate auto update
        if( (_computeOrder & OSGCOMPUTE_UPDATE ) == OSGCOMPUTE_UPDATE )
            setNumChildrenRequiringUpdateTraversal( getNumChildrenRequiringUpdateTraversal() - 1 );

        _computeOrder = co;
        _computeOrderNum = orderNum;

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
            for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
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
    void Computation::addBin( osgUtil::CullVisitor& cv )
    {
        if( !cv.getState() )
        {
            osg::notify(osg::WARN)  << __FUNCTION__ << ": for \""
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
                << getName() << __FUNCTION__ << ": current CullVisitor has no active RenderBin."
                << std::endl;

            return;
        }

        ComputationBin* pb = new ComputationBin;
        if( !pb )
        {
            osg::notify(osg::FATAL)  
                << getName() << __FUNCTION__ << ": cannot create ComputationBin."
                << std::endl;

            return;
        }
        pb->setup( *this );

        //////////////
        // TRAVERSE //
        //////////////
        cv.setCurrentRenderBin( pb );
        cv.apply( *this );
        cv.setCurrentRenderBin( oldRB );

        osgUtil::RenderStage* rs = oldRB->getStage();
        if( rs )
        {
            if( (_computeOrder & OSGCOMPUTE_POSTRENDER) ==  OSGCOMPUTE_POSTRENDER )
            {
                rs->addPostRenderStage(pb,_computeOrderNum);
            }
            else
            {
                rs->addPreRenderStage(pb,_computeOrderNum);
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::launch()
    {            
        // Check if graphics context exist
        // or return otherwise
        if( NULL != GLMemory::getContext() && GLMemory::getContext()->isRealized() )
        {       
            // Launch programs
            if( _launchCallback.valid() ) 
            {
                (*_launchCallback)( *this ); 
            }
            else
            {
                for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
                {
                    if( (*itr)->isEnabled() )
                    {
                        (*itr)->launch();
                    }
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void Computation::applyVisitorToPrograms( osg::NodeVisitor& nv )
    {
        if( nv.getVisitorType() == osg::NodeVisitor::EVENT_VISITOR )
        {
            for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
            {
                if( (*itr)->getEventCallback() )
                    (*(*itr)->getEventCallback())( *(*itr), nv );
            }
        }
        else if( nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR )
        {
            for( ProgramListItr itr = _programs.begin(); itr != _programs.end(); ++itr )
            {
                if( (*itr)->getUpdateCallback() )
                    (*(*itr)->getUpdateCallback())( *(*itr), nv );
            }
        }
    }  
} 