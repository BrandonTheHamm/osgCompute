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
#include <osg/GraphicsContext>
#include <osgCompute/Context>
#include <osgCompute/Resource>

namespace osgCompute
{
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	static OpenThreads::Mutex	s_sharedMutex;
	static std::set< Context* >	s_Contexts;
	static unsigned int			s_ContextIds = 0;

	//------------------------------------------------------------------------------
	static void addContext( Context& context )
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_sharedMutex);

		for( std::set< Context* >::iterator itr = s_Contexts.begin();
			  itr != s_Contexts.end();
			  ++itr )
			if( (*itr) == &context )
				return;

		s_Contexts.insert( &context );
	}

	//------------------------------------------------------------------------------
	static void removeContext( Context& context )
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_sharedMutex);

		for( std::set< Context* >::iterator itr = s_Contexts.begin();
			itr != s_Contexts.end();
			++itr )
			if( (*itr) == &context )
			{
				s_Contexts.erase( itr );
				return;
			}
	}

	//------------------------------------------------------------------------------
	const Context* Context::getContext( unsigned int id )
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_sharedMutex);

		for( std::set< Context* >::iterator itr = s_Contexts.begin();
			itr != s_Contexts.end();
			++itr )
			if( (*itr)->getId() == id )
				return (*itr);

		return NULL;
	}

	//------------------------------------------------------------------------------
	static unsigned int getUniqueContextId()
	{
		return s_ContextIds++;
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Context::Context()
        : osg::Referenced()
    {
		clearLocal();
		_id = getUniqueContextId();
		addContext( *this );
    }

    //------------------------------------------------------------------------------
    Context::~Context()
    {
        clearLocal();
		removeContext( *this );
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

		if( getState() != NULL )
		{
			// if this context is connected to a graphics context 
			// we have to ensure the context is active.
			osg::GraphicsContext* gc = getState()->getGraphicsContext();
			if( gc != NULL && !gc->isCurrent() )
				gc->makeCurrent();
		}
    }

	//------------------------------------------------------------------------------
	unsigned int Context::getId() const
	{
		return _id;
	}

	//------------------------------------------------------------------------------
	void Context::connectToState( osg::State& state )
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
        // do not clear the id!!!
        _state = NULL;
        _clear = true;
        _embedded = false;
        // default use device 0
        _device = 0;
    }
}
