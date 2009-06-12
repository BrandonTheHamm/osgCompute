#include <osg/Node>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osgCompute/Visitor>
#include <osgCompute/Computation>
#include <osgCompute/Resource>

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
                << "osgCompute::ResourceVisitor::init(): no computation applied."
                << std::endl;

            return false;
        }

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::setupForTraversal()
    {
        if( isDirty() )
            return;

        // swap stacks for new traversal
        std::swap( _ptrResources, _ptrOldResources );

        // clear old resource stack
        _ptrResources->clear();
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::updateComputation()
    {
        if( isDirty() )
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
        if( isDirty() )
            return;

        // collect state attributes
        osg::StateSet* ss = node.getStateSet();
        if( NULL != ss )
        {
            osg::StateSet::AttributeList& attr = ss->getAttributeList();
            for( osg::StateSet::AttributeList::iterator itr = attr.begin(); itr != attr.end(); ++itr )
                if( Resource* res = dynamic_cast<Resource*>( (*itr).second.first.get() ) )
                    addResource( *res );

            osg::StateSet::TextureAttributeList& texAttrList = ss->getTextureAttributeList();
            for( osg::StateSet::TextureAttributeList::iterator itr = texAttrList.begin(); itr != texAttrList.end(); ++itr )
            {
                osg::StateSet::AttributeList& texAttr = (*itr);
                for( osg::StateSet::AttributeList::iterator texitr = texAttr.begin(); texitr != texAttr.end(); ++texitr )
                    if( Resource* res = dynamic_cast<Resource*>( (*texitr).second.first.get() ) )
                        addResource( *res );
            }
        }
            
        osg::NodeVisitor::apply( node );
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::apply( osg::Geode& geode )
    {
        if( isDirty() )
            return;

        // collect drawables
        for( unsigned int d=0; d<geode.getNumDrawables(); ++d )
        {
            if( Resource* res = dynamic_cast<Resource*>( geode.getDrawable(d) ) )
                addResource( *res );
        }

        osg::NodeVisitor::apply( geode );
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::apply( osg::Group& group )
    {
        if( isDirty() )
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
        if( isDirty() )
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
        _dirty = true;
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
                << "osgCompute::ContextVisitor::init(): no context applied."
                << std::endl;

            return false;
        }

        _dirty = false;
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
        if( isDirty() )
            return;

        // setup shared context
        if( computation.getContext( _context->getId() ) != _context.get()  )
            computation.setContext( *_context );

        osg::NodeVisitor::apply( computation );
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
        _dirty = true;
    }
}