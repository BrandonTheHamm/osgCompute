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
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int			Resource::s_CurrentIdx = 0;

    //------------------------------------------------------------------------------
    void Resource::setCurrentIdx( unsigned int idx )
    {
        s_CurrentIdx = idx;
    }

    //------------------------------------------------------------------------------
    unsigned int Resource::getCurrentIdx()
    {
        return s_CurrentIdx;
    }

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

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void Resource::addIdentifier( const std::string& handle )
    {
        if( !isAddressedByIdentifier(handle) )
            _identifiers.insert( handle ); 
    }

    //------------------------------------------------------------------------------
    void Resource::removeIdentifier( const std::string& handle )
    {
        IdentifierSetItr itr = _identifiers.find( handle ); 
        if( itr != _identifiers.end() )
            _identifiers.erase( itr );
    }

    //------------------------------------------------------------------------------
    bool Resource::isAddressedByIdentifier( const std::string& handle ) const
    {
        IdentifierSetCnstItr itr = _identifiers.find( handle ); 
        if( itr == _identifiers.end() )
            return false;

        return true;
    }

    //------------------------------------------------------------------------------
    void Resource::setIdentifiers( IdentifierSet& handles )
    {
        _identifiers = handles;
    }

    //------------------------------------------------------------------------------
    IdentifierSet& Resource::getIdentifiers()
    {
        return _identifiers;
    }

    //------------------------------------------------------------------------------
    const IdentifierSet& Resource::getIdentifiers() const
    {
        return _identifiers;
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

    //------------------------------------------------------------------------------
    void Resource::clearCurrent()
    {

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
        _clear = true;
        _identifiers.clear();
    } 


} 
