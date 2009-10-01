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
        if( !isClear() )
            return true;

        if( _handles.empty() )
        {
            addHandle( "notaddressed" );
        }

        _clear = false;
        return true;
    }

	//------------------------------------------------------------------------------
	void Resource::setUpdateResourceCallback( ResourceCallback* uc ) 
	{ 
		_updateCallback = uc; 
	}

	//------------------------------------------------------------------------------
	ResourceCallback* Resource::getUpdateResourceCallback() 
	{ 
		return _updateCallback.get(); 
	}

	//------------------------------------------------------------------------------
	const ResourceCallback* Resource::getUpdateResourceCallback() const 
	{ 
		return _updateCallback.get(); 
	}

	//------------------------------------------------------------------------------
	void Resource::setEventResourceCallback( ResourceCallback* ec ) 
	{ 
		_eventCallback = ec; 
	}

	//------------------------------------------------------------------------------
	ResourceCallback* Resource::getEventResourceCallback() 
	{ 
		return _eventCallback.get(); 
	}

	//------------------------------------------------------------------------------
	const ResourceCallback* Resource::getEventResourceCallback() const 
	{ 
		return _eventCallback.get(); 
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
	bool Resource::isAddressedByHandle( const std::string& handle )
	{
		HandleSetItr itr = _handles.find( handle ); 
		if( itr == _handles.end() )
			return false;

		return true;
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

            if( (*itr) != NULL )
                clear( *(*itr) );
        }
        _contexts.clear();
        _updateCallback = NULL;
        _eventCallback = NULL;
        _clear = true;
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
