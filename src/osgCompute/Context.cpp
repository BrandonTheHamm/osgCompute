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
        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void Context::apply()
    {
        if(isDirty())
            init();
    }

    //------------------------------------------------------------------------------
    void Context::clear()
    {
        clearLocal();
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
        _dirty = true;
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
}
