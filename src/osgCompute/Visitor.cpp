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
    void ResourceVisitor::apply( osg::Node& node )
    {
        if( _currentMode == NONE )
        {
            /////////////
            // COLLECT //
            /////////////
            if( _mode & COLLECT )
            {
                _currentMode = COLLECT;
                collect(node);
                osg::NodeVisitor::traverse( node );
            }
            ////////////////
            // DISTRIBUTE //
            ////////////////
            if( _mode & DISTRIBUTE )
            {
                _currentMode = DISTRIBUTE;
                distribute(node);
                osg::NodeVisitor::traverse( node );
            }
            //////////////
            // EXCHANGE //
            //////////////
            if( _mode & EXCHANGE )
            {
                _currentMode = EXCHANGE;
                exchange(node);
                osg::NodeVisitor::traverse( node );
            }
            ////////////
            // FINISH //
            ////////////
            _currentMode = NONE;
            if( _mode & RESET )
            {
                reset();
            }
        }
        else
        {
            switch( _currentMode )
            {
            case COLLECT:
                collect(node);
                break;
            case DISTRIBUTE:
                distribute(node);
                break;
            case EXCHANGE:
                distribute(node);
                break;
            }

            osg::NodeVisitor::traverse( node );
        }
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::collect( osg::Node& node )
    {
        //////////////////////
        // STATE ATTRIBUTES //
        //////////////////////
        osg::StateSet* ss = node.getStateSet();
        if( NULL != ss )
        {
            osg::StateSet::AttributeList& attr = ss->getAttributeList();
            for( osg::StateSet::AttributeList::iterator itr = attr.begin(); itr != attr.end(); ++itr )
                if( InteropObject* res = dynamic_cast<InteropObject*>( (*itr).second.first.get() ) )
                {
                    osg::ref_ptr<Resource> resource = res->getOrCreateInteropMemory();
                    addResource( *res->getOrCreateInteropMemory() );
                }

            osg::StateSet::TextureAttributeList& texAttrList = ss->getTextureAttributeList();
            for( osg::StateSet::TextureAttributeList::iterator itr = texAttrList.begin(); itr != texAttrList.end(); ++itr )
            {
                osg::StateSet::AttributeList& texAttr = (*itr);
                for( osg::StateSet::AttributeList::iterator texitr = texAttr.begin(); texitr != texAttr.end(); ++texitr )
                {
                    if( InteropObject* res = dynamic_cast<InteropObject*>( (*texitr).second.first.get() ) )
                    {
                        osg::ref_ptr<Resource> resource = res->getOrCreateInteropMemory();
                        addResource( *resource );
                    }
                }
            }
        }

        ///////////////
        // DRAWABLES //
        ///////////////
        osg::Geode* geode = dynamic_cast<osg::Geode*>( &node );
        if( NULL != geode )
        {
            for( unsigned int d=0; d<geode->getNumDrawables(); ++d )
            {
                if( InteropObject* res = dynamic_cast<InteropObject*>( geode->getDrawable(d) ) )
                {
                    osg::ref_ptr<Resource> resource = res->getOrCreateInteropMemory();
                    addResource( *resource );
                }
            }
        }

        //////////////
        // INTERNAL //
        //////////////
        osgCompute::Computation* computation = dynamic_cast<osgCompute::Computation*>( &node );
        if( NULL != computation )
        {
            ResourceHandleList& resources = computation->getResources();
            for( ResourceHandleListItr itr = resources.begin(); itr != resources.end(); ++itr )
            {
                if( (*itr)._resource.valid() )
                {
                    addResource( *((*itr)._resource) );
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::exchange( osg::Node& node )
    {
        //////////////
        // INTERNAL //
        //////////////
        osgCompute::Computation* computation = dynamic_cast<osgCompute::Computation*>( &node );
        if( NULL != computation )
        {
            for( ResourceSetItr itr = _collectedResources.begin(); itr != _collectedResources.end(); ++itr )
            {
                if( (*itr).valid() )
                {
                    computation->exchangeResource( *(*itr), false );
                }
            }
        }
    }


    //------------------------------------------------------------------------------
    void ResourceVisitor::distribute( osg::Node& node )
    {
        //////////////
        // INTERNAL //
        //////////////
        osgCompute::Computation* computation = dynamic_cast<osgCompute::Computation*>( &node );
        if( NULL != computation )
        {
            for( ResourceSetItr itr = _collectedResources.begin(); itr != _collectedResources.end(); ++itr )
            {
                if( (*itr).valid() )
                {
                    computation->addResource( *(*itr), false );
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::addResource( Resource& resource )
    {
        osg::ref_ptr<Resource> tmpResource = &resource;
        if( !hasResource(resource) )
            _collectedResources.insert( &resource );
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::removeResource( Resource& resource )
    {
        osg::ref_ptr<Resource> tmpResource = &resource;
        ResourceSetItr itr = _collectedResources.find( tmpResource );
        if( itr != _collectedResources.end() )
            _collectedResources.erase( itr );
    }

    //------------------------------------------------------------------------------
    bool ResourceVisitor::hasResource( Resource& resource )
    {
        osg::ref_ptr<Resource> ptr = &resource;
        ResourceSetItr itr = _collectedResources.find( ptr );
        if( itr != _collectedResources.end() )
            return true;

        return false;
    }

    //------------------------------------------------------------------------------
    ResourceSet& ResourceVisitor::getResources()
    {
        return _collectedResources;
    }

    //------------------------------------------------------------------------------
    const ResourceSet& ResourceVisitor::getResources() const
    {
        return _collectedResources;
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::setMode( unsigned int mode )
    {
        _mode = mode;
    }

    //------------------------------------------------------------------------------
    unsigned int ResourceVisitor::getMode()
    {
        return _mode;
    }

    //------------------------------------------------------------------------------
    void ResourceVisitor::reset() 
    {
        _collectedResources.clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ResourceVisitor::clearLocal()
    {
        _currentMode = NONE;
        _mode = COLLECT | DISTRIBUTE | RESET;
        reset();
    }
} 
