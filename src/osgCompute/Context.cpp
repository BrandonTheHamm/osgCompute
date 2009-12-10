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
	const Context* Context::getContextFromGraphicsContext( unsigned int graphicContextId )
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_sharedMutex);

		for( std::set< Context* >::iterator itr = s_Contexts.begin();
			itr != s_Contexts.end();
			++itr )
			if( (*itr)->isConnectedWithGraphicsContext() && 
				(*itr)->getGraphicsContext()->getState()->getContextID() == graphicContextId )
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
            return;

		// if this context is connected to a graphics context 
		// we have to ensure the context is active.
		if( _gc.valid() && !_gc->isCurrent() )
			_gc->makeCurrent();
    }

	//------------------------------------------------------------------------------
	unsigned int Context::getId() const
	{
		return _id;
	}

	//------------------------------------------------------------------------------
	void Context::connectWithGraphicsContext( osg::GraphicsContext& gc )
	{
		if( !isClear() )
			return;

		_gc = &gc;
	}

	//------------------------------------------------------------------------------
	osg::GraphicsContext* Context::getGraphicsContext()
	{
		return _gc.get();
	}

	//------------------------------------------------------------------------------
	const osg::GraphicsContext* Context::getGraphicsContext() const
	{
		return _gc.get();
	}

	//------------------------------------------------------------------------------
	void Context::removeGraphicsContext()
	{
		if( !isClear() )
			return;

		_gc = NULL;
	}

	//------------------------------------------------------------------------------
	bool Context::isConnectedWithGraphicsContext() const
	{
		return _gc.valid();
	}

	//------------------------------------------------------------------------------
	void Context::setDevice( int device )
	{ 
		if( !isClear() )
			return;

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
	void Context::clearResources() const
	{
		while( !_resources.empty() )
		{
			ResourcePtrSetItr itr = _resources.begin();
			(*itr)->clear( *this );
		}
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
		// clear resources first!!!
		clearResources();
        // do not clear the id!!!
        _gc = NULL;
        _clear = true;
        _embedded = false;
        // default use device 0
        _device = 0;
    }

	//------------------------------------------------------------------------------
	void Context::registerResource( const Resource& resource ) const
	{
		ResourcePtrSetItr itr = _resources.find( &resource );
		if( itr != _resources.end() )
			return;

		_resources.insert( &resource );
	}

	//------------------------------------------------------------------------------
	void Context::unregisterResource( const Resource& resource ) const
	{
		ResourcePtrSetItr itr = _resources.find( &resource );
		if( itr == _resources.end() )
			return;

		_resources.erase( itr );
	}
}
