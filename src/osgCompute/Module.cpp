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

#include <osgCompute/Module>
#include <osgCompute/Context>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clear()
    {
        clearLocal();
        Resource::clear();
    }

    //------------------------------------------------------------------------------
    bool Module::init()
    {
        return Resource::init();
    }

    //------------------------------------------------------------------------------
    void Module::acceptResource( Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    bool Module::usesResource( const std::string& handle ) const 
    { 
        return false; 
    }

    //------------------------------------------------------------------------------
    void Module::removeResource( const std::string& handle ) 
    {
    }

    //------------------------------------------------------------------------------
    void Module::removeResource( const Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    Resource* Module::getResource( const std::string& handle ) 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    const Resource* Module::getResource( const std::string& handle ) const 
    { 
        return NULL; 
    }

	//------------------------------------------------------------------------------
	void Module::setLaunchCallback( LaunchCallback* pc ) 
	{ 
		_launchCallback = pc; 
	}

	//------------------------------------------------------------------------------
	LaunchCallback* Module::getLaunchCallback() 
	{ 
		return _launchCallback; 
	}

	//------------------------------------------------------------------------------
	const LaunchCallback* Module::getLaunchCallback() const 
	{ 
		return _launchCallback; 
	}

	//------------------------------------------------------------------------------
	void Module::enable() 
	{ 
		_enabled = true; 
	}

	//------------------------------------------------------------------------------
	void Module::disable() 
	{ 
		_enabled = false; 
	}

	//------------------------------------------------------------------------------
	bool Module::isEnabled() const
	{
		return _enabled;
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clearLocal()
    {
        _launchCallback = NULL;
        _enabled = true;
        _clear = true;
    }

    //------------------------------------------------------------------------------
    bool Module::init( const Context& context ) const
    {
        return Resource::init( context );
    }

    //------------------------------------------------------------------------------
    void Module::clear( const Context& context ) const
    {
        if( _launchCallback )
            _launchCallback->clear( context );

        return Resource::clear( context );
    }
}