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
#include <osg/Notify>
#include <osgCompute/Context>
#include <osgCompute/Resource>

namespace osgCompute
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Context::Context()
        : osg::Object()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    Context::~Context()
    {
        clearLocal();
    }


    //------------------------------------------------------------------------------
    bool Context::init()
    {
        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void Context::apply()
    {
        if(isClear())
            init();
    }

	//------------------------------------------------------------------------------
	void Context::setId( unsigned int id )
	{
		if( !isClear() )
			return;

		_id = id;
	}

	//------------------------------------------------------------------------------
	unsigned int Context::getId() const
	{
		return _id;
	}

	//------------------------------------------------------------------------------
	void Context::setState( osg::State& state )
	{
		if( !isClear() )
			return;

		_state = &state;
	}

	//------------------------------------------------------------------------------
	osg::State* Context::getState()
	{
		return _state.get();
	}

	//------------------------------------------------------------------------------
	const osg::State* Context::getState() const
	{
		return _state.get();
	}

	//------------------------------------------------------------------------------
	void Context::removeState()
	{
		_state = NULL;
	}

	//------------------------------------------------------------------------------
	bool Context::isStateValid() const
	{
		return _state.valid();
	}

	//------------------------------------------------------------------------------
	void Context::setDevice( int device )
	{ 
		_device = device;  
	}

	//------------------------------------------------------------------------------
	int Context::getDevice() const 
	{ 
		return _device; 
	}

	//------------------------------------------------------------------------------
	bool Context::isClear() const
	{
		return _clear;
	}

    //------------------------------------------------------------------------------
    void Context::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Context::clearLocal()
    {
        // free context dependent memory
        clearResources();

        // do not clear the id!!!
        _state = NULL;
        _clear = true;
        _embedded = false;
        // default use device 0
        _device = 0;
    }

    //------------------------------------------------------------------------------
    void Context::clearResources() const
    {
        while( !_resources.empty() )
        {
            Resource* curResource = _resources.front();

            // each of the resources calls unregisterResource()
            // within the clear(\"CONTEXT\") function
            if( curResource )
                curResource->clear( *this );
        }
    }

	//------------------------------------------------------------------------------
	bool osgCompute::Context::isRegistered( Resource& resource ) const
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

		for( ResourceListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
			if( (*itr) == &resource )
				return true;

		return false;
	}

	//------------------------------------------------------------------------------
	void Context::registerResource( Resource& resource ) const
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

		_resources.push_back( &resource );
	}

	//------------------------------------------------------------------------------
	void Context::unregisterResource( Resource& resource ) const
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

		for( ResourceListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
		{
			if( (*itr) == &resource )
			{
				_resources.erase( itr );
				return;
			}
		}
	}
}
