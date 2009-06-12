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
        if( !isDirty() )
            return true;

        if( _handles.empty() )
        {
            addHandle( "notaddressed" );
        }

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    unsigned int osgCompute::Resource::getByteSize() const
    {
        return 0;
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

            if( (*itr) != NULL )
                clear( *(*itr) );
        }
        _contexts.clear();
        _updateCallback = NULL;
        _eventCallback = NULL;
        _dirty = true;
    } 

    //------------------------------------------------------------------------------
    bool Resource::init( const Context& context ) const
    {
        if( context.isRegistered(const_cast<Resource&>(*this)) )
            return true;

        context.registerResource(const_cast<Resource&>(*this));

        ContextSetCnstItr itr = _contexts.find( &context );
        if( itr == _contexts.end() || (*itr) == NULL )
            _contexts.insert( &context );

        return true;
    }

    //------------------------------------------------------------------------------
    void Resource::clear( const Context& context ) const
    {
        context.unregisterResource(const_cast<Resource&>(*this));

        ContextSetItr itr = _contexts.find( &context );
        if( itr != _contexts.end() && (*itr) != NULL )
            _contexts.erase( itr );
    }

    //------------------------------------------------------------------------------
    const Context* Resource::getContext( unsigned int ctxId ) const
    {
        for( ContextSetCnstItr itr = _contexts.begin(); itr != _contexts.end(); ++itr )
            if( (*itr) != NULL && (*itr)->getId() == ctxId )
                return (*itr);

        return NULL;
    }
}
