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
#include "osgCompute/Param"

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool Param::init()
    {
        if( !isDirty() )
            return true;

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void Param::clear()
    {
        clearLocal();
        ContextResource::clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Param::clearLocal()
    {
        _subloadCallback = NULL;
        _updateCallback = NULL;
        _eventCallback = NULL;
        _dirty = true;
    }

    //------------------------------------------------------------------------------
    bool Param::init( const Context& context ) const
    {
        return ContextResource::init( context );
    }
    
    //------------------------------------------------------------------------------
    void Param::clear( const Context& context ) const
    {
        return ContextResource::clear( context );
    }
}
