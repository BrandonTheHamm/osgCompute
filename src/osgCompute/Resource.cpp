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

#include <osg/Notify>
#include <osgCompute/Resource>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    osg::ref_ptr<ResourceObserver> ResourceObserver::s_resourceObserver;

    //------------------------------------------------------------------------------
    ResourceObserver* ResourceObserver::instance()
    {
        if( !s_resourceObserver.valid() )
            s_resourceObserver = new ResourceObserver;

        return s_resourceObserver.get();
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ResourceObserver::ResourceObserver() :
    Observer(),
        Referenced()
    {
    }

    //------------------------------------------------------------------------------
    ResourceObserver::~ResourceObserver()
    {
    }


    //------------------------------------------------------------------------------
    void ResourceObserver::observeResource( Resource& resource )
    {
        std::string resourceClassSpecifier = std::string(resource.libraryName()).append("::").append(resource.className());

        if( resource.getObserverSet() != NULL )
        {
            std::set<Observer*>& observers = resource.getObserverSet()->getObservers();
            for( std::set<Observer*>::iterator itr = observers.begin(); itr != observers.end(); ++itr )
            {
                if( (*itr) == this )
                {   // Object is already observed, so change the className and libraryName
                    for( ObserverMapItr obsItr = _observedObjects.begin(); obsItr != _observedObjects.end(); ++obsItr )
                    {
                        if( obsItr->second == &resource )
                        {
                            _observedObjects.erase( obsItr );
                            _observedObjects.insert( std::make_pair<std::string, osg::observer_ptr<Resource> >( resourceClassSpecifier, &resource ) );
                            return;
                        }
                    }

                }
            }
        }

        resource.addObserver( this );
        _observedObjects.insert( std::make_pair<std::string, osg::observer_ptr<Resource> >( resourceClassSpecifier, &resource ) );
    }

    //------------------------------------------------------------------------------
    void ResourceObserver::objectDeleted( void* object )
    {
        Resource* resource = static_cast<Resource*>( object );
        if( resource == NULL ) return;

        std::string resourceClassSpecifier = std::string(resource->libraryName()).append("::").append(resource->className());
        std::pair<ObserverMapItr,ObserverMapItr> range = _observedObjects.equal_range( resourceClassSpecifier );
        while( range.first != range.second )
        {
            if( range.first->second.valid() && range.first->second == resource )
            {
                _observedObjects.erase( range.first );
                return;
            }

            range.first++;
        }
    }

    //------------------------------------------------------------------------------
    ResourceClassList ResourceObserver::getResources( std::string classIdentifier ) const
    {
        ResourceClassList resourceList;
        std::pair<ObserverMapCnstItr,ObserverMapCnstItr> range = _observedObjects.equal_range( classIdentifier );
        for(;range.first != range.second; range.first++ )
        {
            if( range.first->second.valid() )
            {
                osg::observer_ptr<Resource> resObs;
                resObs = (range.first->second);
                resourceList.push_back( resObs );
            }
        }

        return resourceList;
    }

    //------------------------------------------------------------------------------
    void ResourceObserver::releaseAllResourceObjects()
    {
        for( ObserverMapItr itr = _observedObjects.begin(); itr != _observedObjects.end(); ++itr )
        {
            if( (*itr).second.valid() )
                (*itr).second->releaseObjects();
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Resource::Resource()
    {
        _objectsReleased = true;
    }

    //------------------------------------------------------------------------------
    void Resource::addIdentifier( const std::string& handle )
    {
        if( !isIdentifiedBy(handle) )
            _identifiers.insert( handle ); 
    }

    //------------------------------------------------------------------------------
    void Resource::removeIdentifier( const std::string& handle )
    {
        IdentifierSetItr itr = _identifiers.find( handle ); 
        if( itr != _identifiers.end() )
            _identifiers.erase( itr );
    }

    //------------------------------------------------------------------------------
    bool Resource::isIdentifiedBy( const std::string& handle ) const
    {
        IdentifierSetCnstItr itr = _identifiers.find( handle ); 
        if( itr == _identifiers.end() )
            return false;

        return true;
    }

    //------------------------------------------------------------------------------
    void Resource::setIdentifiers( IdentifierSet& handles )
    {
        _identifiers = handles;
    }

    //------------------------------------------------------------------------------
    IdentifierSet& Resource::getIdentifiers()
    {
        return _identifiers;
    }

    //------------------------------------------------------------------------------
    const IdentifierSet& Resource::getIdentifiers() const
    {
        return _identifiers;
    }

    //------------------------------------------------------------------------------
    void Resource::removeAllIdentifiers()
    {
        _identifiers.clear();
    }

    //------------------------------------------------------------------------------
    void Resource::clear()
    {
        releaseObjects();
    }

    //------------------------------------------------------------------------------
    void Resource::releaseObjects()
    {
        _objectsReleased = true;
    }

    //------------------------------------------------------------------------------
    void Resource::releaseGLObjects( osg::State* state )
    {
        // Do release all resources no matter which state is applied.
        // Currently only a single context is supported by osgCompute
        // so there should exist only one state
        if( !objectsReleased() ) releaseObjects();
    }

    //------------------------------------------------------------------------------
    bool Resource::objectsReleased()
    {
        return _objectsReleased;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Resource::~Resource()
    {
        // free a resource's allocations
        releaseObjects();
    }

    //------------------------------------------------------------------------------
    void Resource::objectsCreated() const
    {
        _objectsReleased = false;
    } 

} 
