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

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool Module::init()
    {
        return Resource::init();
    }

    //------------------------------------------------------------------------------
    void Module::setUpdateCallback( ModuleCallback* uc ) 
    { 
        _updateCallback = uc; 
    }

    //------------------------------------------------------------------------------
    ModuleCallback* Module::getUpdateCallback() 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ModuleCallback* Module::getUpdateCallback() const 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Module::setEventCallback( ModuleCallback* ec ) 
    { 
        _eventCallback = ec; 
    }

    //------------------------------------------------------------------------------
    ModuleCallback* Module::getEventCallback() 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ModuleCallback* Module::getEventCallback() const 
    { 
        return _eventCallback.get(); 
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
    void Module::getResources( ResourceList& resourceList, const std::string& handle ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Module::getAllResources( ResourceList& resourceList ) 
    { 
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

    //------------------------------------------------------------------------------
    void Module::clear()
    {
        clearLocal();
        Resource::clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clearLocal()
    {
        _eventCallback = NULL;
        _updateCallback = NULL;
        _enabled = true;
        _clear = true;
    }
}
