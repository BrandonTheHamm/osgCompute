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

#include "osgCompute/Module"

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clear()
    {
        clearLocal();
        ContextResource::clear();
    }

    //------------------------------------------------------------------------------
    bool Module::init()
    {
        if( !isDirty() )
            return true;

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    inline void Module::acceptParam( const std::string& handle, osgCompute::Param& param ) 
    {
    }

    //------------------------------------------------------------------------------
    inline bool Module::usesParam( const std::string& handle ) const 
    { 
        return false; 
    }

    //------------------------------------------------------------------------------
    inline void Module::removeParam( const std::string& handle, const osgCompute::Param* param /*= NULL*/ ) 
    {
    }

    //------------------------------------------------------------------------------
    inline Param* Module::getParam( const std::string& handle ) 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    inline const Param* Module::getParam( const std::string& handle ) const 
    { 
        return NULL; 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clearLocal()
    {
        _launchCallback = NULL;
        _updateCallback = NULL;
        _eventCallback = NULL;
        _enabled = true;
        _dirty = true;
    }

    //------------------------------------------------------------------------------
    bool Module::init( const Context& context ) const
    {
        return ContextResource::init( context );
    }

    //------------------------------------------------------------------------------
    void Module::clear( const Context& context ) const
    {
        if( _launchCallback )
            _launchCallback->clear( context );

        return ContextResource::clear( context );
    }
}