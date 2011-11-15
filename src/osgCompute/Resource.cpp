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
        if( resource.getObserverSet() != NULL )
        {
            std::set<Observer*>& observers = resource.getObserverSet()->getObservers();
            for( std::set<Observer*>::iterator itr = observers.begin(); itr != observers.end(); ++itr )
            {
                if( (*itr) == this )
                    return;// Object is already observed
            }
        }

        resource.addObserver( this );

        std::string resourceClassSpecifier = std::string(resource.libraryName()).append("::").append(resource.className());
        _observedObjects.insert( std::make_pair<std::string, osg::observer_ptr<Resource> >( resourceClassSpecifier, &resource ) );
    }

    //------------------------------------------------------------------------------
    void ResourceObserver::objectDeleted( void* object )
    {
        Resource* resource = static_cast<Resource*>( object );
        if( resource == NULL ) return;

        std::string resourceClassName = resource->className();
        std::pair<ObserverMapItr,ObserverMapItr> range = _observedObjects.equal_range( resourceClassName );
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

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Resource::Resource()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool Resource::init()
    {
        if( !isClear() )
            return true;


        ResourceObserver::instance()->observeResource( *this );
        _clear = false;
        return true;
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
    bool Resource::isClear() const 
    { 
        return _clear; 
    }

    //------------------------------------------------------------------------------
    void Resource::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Resource::releaseObjects()
    {
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Resource::~Resource()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Resource::clearLocal()
    {
        _clear = true;
        _identifiers.clear();
        
        // free memory
        releaseObjects();
    } 


} 
