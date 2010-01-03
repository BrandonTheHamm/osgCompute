#include <osg/Node>
#include <osg/Geode>
#include <osg/Group>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osgCompute/Visitor>
#include <osgCompute/Computation>
#include <osgCompute/Interoperability>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ResourceVisitor::ResourceVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::NODE_VISITOR,osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool ResourceVisitor::init()
    {
        if( NULL == _computation )
        {
            osg::notify(osg::FATAL)  
                << " [osgCompute::ResourceVisitor::init()]: no computation attached."
                << std::endl;

            return false;
        }

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::setupForTraversal()
    {
        if( isClear() )
            return;

        // swap stacks for new traversal
        std::swap( _ptrResources, _ptrOldResources );

        // clear old resource stack
        _ptrResources->clear();
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::updateComputation()
    {
        if( isClear() )
            return;

        // check for removed resources
        ResourceSetItr searchRes = _ptrResources->end();
        for( ResourceSetItr itr = _ptrOldResources->begin(); itr != _ptrOldResources->end(); ++itr )
        {
            searchRes = _ptrResources->find( (*itr) );
            if( searchRes == _ptrResources->end() )
            {
                // resource doesn't exist anymore, so remove it
                getComputation()->removeResource( *(*itr) );
            }
        }

        // check for new resources
        for( ResourceSetItr itr = _ptrResources->begin(); itr != _ptrResources->end(); ++itr )
        {
            searchRes = _ptrOldResources->find( (*itr) );
            if( searchRes == _ptrOldResources->end() )
            {
                // add this resource to the computation
                getComputation()->addResource( *(*itr) );
            }
        }
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::apply( osg::Node& node )
    {
        if( isClear() )
            return;

        // collect state attributes
        osg::StateSet* ss = node.getStateSet();
        if( NULL != ss )
        {
            osg::StateSet::AttributeList& attr = ss->getAttributeList();
            for( osg::StateSet::AttributeList::iterator itr = attr.begin(); itr != attr.end(); ++itr )
                if( InteropObject* res = dynamic_cast<InteropObject*>( (*itr).second.first.get() ) )
                    addResource( *res->getOrCreateBuffer() );

            osg::StateSet::TextureAttributeList& texAttrList = ss->getTextureAttributeList();
            for( osg::StateSet::TextureAttributeList::iterator itr = texAttrList.begin(); itr != texAttrList.end(); ++itr )
            {
                osg::StateSet::AttributeList& texAttr = (*itr);
                for( osg::StateSet::AttributeList::iterator texitr = texAttr.begin(); texitr != texAttr.end(); ++texitr )
                    if( InteropObject* res = dynamic_cast<InteropObject*>( (*texitr).second.first.get() ) )
                        addResource( *res->getOrCreateBuffer() );
            }
        }
            
        osg::NodeVisitor::apply( node );
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::apply( osg::Geode& geode )
    {
        if( isClear() )
            return;

        // collect drawables
        for( unsigned int d=0; d<geode.getNumDrawables(); ++d )
        {
            if( InteropObject* res = dynamic_cast<InteropObject*>( geode.getDrawable(d) ) )
                addResource( *res->getOrCreateBuffer() );
        }

        osg::NodeVisitor::apply( geode );
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::apply( osg::Group& group )
    {
        if( isClear() )
            return;

        Computation* computation = dynamic_cast<Computation*>( &group );
        if( computation != NULL )
            apply( *computation );
        else
            osg::NodeVisitor::apply( group );
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::apply( Computation& computation )
    {
        if( isClear() )
            return;

        // computation has started its own traversal 

        ResourceMap& resources = computation.getResources();
        for( ResourceMapItr itr = resources.begin(); itr != resources.end(); ++itr )
            if( (*itr).first != NULL )
                addResource( *((*itr).first) );

        // mark computation as child computation
        computation.setParentComputation( getComputation() );

        // do not traverse the subgraph as it is
        // done by the computation itself
    }

	//------------------------------------------------------------------------------
	bool ResourceVisitor::isClear() const
	{
		return _clear;
	}

	//------------------------------------------------------------------------------
	void ResourceVisitor::setComputation( Computation* computation )
	{
		_computation = computation;
	}

	//------------------------------------------------------------------------------
	Computation* ResourceVisitor::getComputation()
	{
		return _computation;
	}

	//------------------------------------------------------------------------------
	const Computation* ResourceVisitor::getComputation() const
	{
		return _computation;
	}

	//------------------------------------------------------------------------------
	void ResourceVisitor::addResource( Resource& resource )
	{
		_ptrResources->insert( &resource );
	}

	//------------------------------------------------------------------------------
	void ResourceVisitor::removeResource( Resource& resource )
	{
		ResourceSetItr itr = _ptrResources->find( &resource );
		if( itr != _ptrResources->end() )
			_ptrResources->erase( itr );
	}

	//------------------------------------------------------------------------------
	bool ResourceVisitor::hasResource( Resource& resource )
	{
		ResourceSetItr itr = _ptrResources->find( &resource );
		if( itr != _ptrResources->end() )
			return true;

		return false;
	}

	//------------------------------------------------------------------------------
	ResourceSet& ResourceVisitor::getResources()
	{
		return *_ptrResources;
	}

	//------------------------------------------------------------------------------
	const ResourceSet& ResourceVisitor::getResources() const
	{
		return *_ptrResources;
	}

    //------------------------------------------------------------------------------
    void ResourceVisitor::reset() 
    {
        clear();
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ResourceVisitor::clearLocal()
    {
        _computation = NULL;
        _resourcesA.clear();
        _resourcesB.clear();

        _ptrResources = &_resourcesA;
        _ptrOldResources = &_resourcesB;
        _clear = true;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ContextVisitor::ContextVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::NODE_VISITOR,osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool ContextVisitor::init()
    {
        if( !_context.valid() )
        {
            osg::notify(osg::FATAL)  
                << " [osgCompute::ContextVisitor::init()]: no context attached."
                << std::endl;

            return false;
        }

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void ContextVisitor::reset() 
    {
        clear();
    }

    //------------------------------------------------------------------------------
    void ContextVisitor::apply( Computation& computation )
    {
        if( isClear() )
            return;

        // setup shared context
		osg::GraphicsContext* gc = _context->getGraphicsContext();
        if( computation.getContext( *gc->getState() ) != _context.get()  )
            computation.acceptContext( *_context );

        osg::NodeVisitor::traverse( computation );
    }

	//------------------------------------------------------------------------------
	void ContextVisitor::apply( osg::Group& group )
	{
		if( Computation* comp = dynamic_cast<Computation*>( &group ) )
			apply( *comp );
		else
			osg::NodeVisitor::apply( group );
	}

	//------------------------------------------------------------------------------
	void ContextVisitor::setContext( Context* context )
	{
		_context = context;
	}

	//------------------------------------------------------------------------------
	Context* ContextVisitor::getContext()
	{
		return _context.get();
	}

	//------------------------------------------------------------------------------
	const Context* ContextVisitor::getContext() const
	{
		return _context.get();
	}

	//------------------------------------------------------------------------------
	bool ContextVisitor::isClear() const
	{
		return _clear;
	}

    //------------------------------------------------------------------------------
    void ContextVisitor::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ContextVisitor::clearLocal()
    {
        _context = NULL;
        _clear = true;
    }
}