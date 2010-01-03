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
#include <osgCompute/Context>
#include <osgCompute/Resource>

namespace osgCompute
{   
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

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void Resource::addHandle( const std::string& handle )
    {
        if( !isAddressedByHandle(handle) )
            _handles.insert( handle ); 
    }

    //------------------------------------------------------------------------------
    void Resource::removeHandle( const std::string& handle )
    {
        HandleSetItr itr = _handles.find( handle ); 
        if( itr != _handles.end() )
            _handles.erase( itr );
    }

    //------------------------------------------------------------------------------
    bool Resource::isAddressedByHandle( const std::string& handle ) const
    {
        HandleSetCnstItr itr = _handles.find( handle ); 
        if( itr == _handles.end() )
            return false;

        return true;
    }

    //------------------------------------------------------------------------------
    HandleSet& Resource::getHandles()
    {
        return _handles;
    }

    //------------------------------------------------------------------------------
    const HandleSet& Resource::getHandles() const
    {
        return _handles;
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
        while( !_contexts.empty() )
        {
            ContextSetItr itr = _contexts.begin();
            clear( *(*itr) );
        }

        _clear = true;
        _handles.clear();
    } 

    //------------------------------------------------------------------------------
    void Resource::init( const Context& context ) const
    {
        ContextSetCnstItr itr = _contexts.find( &context );
        if( itr != _contexts.end() )
            return;

        _contexts.insert( &context );
        context.registerResource( *this );
    }

    //------------------------------------------------------------------------------
    void Resource::clear( const Context& context ) const
    {
        ContextSetItr itr = _contexts.find( &context );
        if( itr == _contexts.end() )
            return;

        _contexts.erase( itr );
        context.unregisterResource( *this );
    }

    //------------------------------------------------------------------------------
    void Resource::setHandles( HandleSet& handles )
    {
        _handles = handles;
    }
} 
