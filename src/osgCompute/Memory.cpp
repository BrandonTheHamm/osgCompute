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
#include <osgCompute/Memory>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    MemoryObject::MemoryObject() 
        :   Referenced(),
            _mapping( UNMAPPED ),
            _allocHint(0),
            _pitch(0),
            _syncOp(NO_SYNC)
    {
    }

    //------------------------------------------------------------------------------
    MemoryObject::~MemoryObject() 
    {
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Memory::Memory() 
        : Resource()
    { 
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Memory::clear()
    {
        Resource::clear();
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Memory::clearCurrent()
    {
        if( Resource::getCurrentIdx() < _objects.size() )
            _objects[ Resource::getCurrentIdx() ] = NULL;
    }

    //------------------------------------------------------------------------------
    bool Memory::init()
    {
        if( !isClear() )
            return true;

        if( _dimensions.empty() )
        {
            osg::notify(osg::FATAL)  
                << getName() << " [Memory::init()]: no dimensions specified."                  
                << std::endl;

            return false;
        }

        if( _elementSize == 0 )
        {
            osg::notify(osg::FATAL)  
                << getName() << " [Memory::init()]: no element size specified."                  
                << std::endl;

            return false;
        }

        ///////////////////////
        // COMPUTE BYTE SIZE //
        ///////////////////////
        _numElements = 1;
        for( unsigned int d=0; d<_dimensions.size(); ++d )
            _numElements *= _dimensions[d];

        return Resource::init();
    }

    //------------------------------------------------------------------------------
    void Memory::setElementSize( unsigned int elementSize ) 
    { 
        if( !isClear() )
            return;

        _elementSize = elementSize; 
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getElementSize() const 
    { 
        return _elementSize; 
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getByteSize() const 
    { 
        return getElementSize() * getNumElements(); 
    }

    //------------------------------------------------------------------------------
    void Memory::setDimension( unsigned int dimIdx, unsigned int dimSize )
    {
        if( !isClear() )
            return;

        if (_dimensions.size()<=dimIdx)
            _dimensions.resize(dimIdx+1,0);

        _dimensions[dimIdx] = dimSize;
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getDimension( unsigned int dimIdx ) const
    { 
        if( dimIdx > (_dimensions.size()-1) )
            return 0;

        return _dimensions[dimIdx];
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getNumDimensions() const
    { 
        return _dimensions.size();
    }

    //------------------------------------------------------------------------------
    unsigned int osgCompute::Memory::getNumElements() const
    {
        return _numElements;
    }

    //------------------------------------------------------------------------------
    void osgCompute::Memory::setAllocHint( unsigned int allocHint )
    {
        if( !isClear() )
            return;

        _allocHint = (_allocHint | allocHint);
    }

    //------------------------------------------------------------------------------
    unsigned int osgCompute::Memory::getAllocHint() const
    {
        return _allocHint;
    }

    //------------------------------------------------------------------------------
    void Memory::setSubloadCallback( SubloadCallback* sc ) 
    { 
        _subloadCallback = sc; 
    }

    //------------------------------------------------------------------------------
    SubloadCallback* Memory::getSubloadCallback() 
    { 
        return _subloadCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const SubloadCallback* Memory::getSubloadCallback() const 
    { 
        return _subloadCallback.get(); 
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getMapping( unsigned int ) const
    {
        if( isClear() )
            return osgCompute::UNMAPPED;

        if( Resource::getCurrentIdx() > _objects.size() )
            return osgCompute::UNMAPPED;

        if( _objects[ Resource::getCurrentIdx() ].valid() )
            return _objects[ Resource::getCurrentIdx() ]->_mapping;
        else 
            return osgCompute::UNMAPPED;
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getPitch( unsigned int hint /*= 0 */ ) const
    {
        if( isClear() )
            return 0;

        if( _pitch == 0 )
            _pitch = computePitch();
                
        return _pitch;
    }


    //------------------------------------------------------------------------------
    void Memory::swap( unsigned int incr /*= 1 */ )
    {
        // Function should be implemented by swap buffers.
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getSwapCount() const
    {
        // Function should be implemented by swap buffers.
        return 0;
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getSwapIdx() const
    {
        // Function should be implemented by swap buffers.
        return 0;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Memory::~Memory() 
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------
    void Memory::clearLocal()
    {
        _dimensions.clear();
        _numElements = 0;
        _elementSize = 0;
        _allocHint = 0;
        _subloadCallback = NULL;
        _pitch = 0;
        _objects.clear(); // free memory
    }


    //------------------------------------------------------------------------------
    MemoryObject* Memory::object()
    {
        if( isClear() )
            return NULL;

        // resize memory array for new context
        if( (Resource::getCurrentIdx()+1) > _objects.size() )
            _objects.resize( Resource::getCurrentIdx()+1 );

        // Allocate/Register handles on demand
        if( !_objects[Resource::getCurrentIdx()].valid() )
        {
            MemoryObject* newObject = createObject();
            if( newObject == NULL )
            {
                osg::notify( osg::FATAL )  
                    << getName() << " [Memory::getObject()] \"" << getName() << "\": allocation of memory failed."
                    << std::endl;
                return NULL;
            }
            newObject->_mapping = osgCompute::UNMAPPED;
            newObject->_allocHint = getAllocHint();
            _objects[Resource::getCurrentIdx()] = newObject;
        }

        return _objects[Resource::getCurrentIdx()].get();
    }

    //------------------------------------------------------------------------------
    const MemoryObject* Memory::object() const
    {
        if( isClear() )
            return NULL;

        // resize memory array for new context
        if( (Resource::getCurrentIdx()+1) > _objects.size() )
            _objects.resize( Resource::getCurrentIdx()+1 );

        // Allocate/Register handles on demand
        if( !_objects[Resource::getCurrentIdx()].valid() )
        {
            MemoryObject* newObject = createObject();
            if( newObject == NULL )
            {
                osg::notify( osg::FATAL )  
                    << getName() << " [Memory::getObject()] \"" << getName() << "\": allocation of memory failed."
                    << std::endl;
                return NULL;
            }
            newObject->_mapping = osgCompute::UNMAPPED;
            newObject->_allocHint = getAllocHint();
            _objects[Resource::getCurrentIdx()] = newObject;
        }

        return _objects[Resource::getCurrentIdx()].get();
    }

    //------------------------------------------------------------------------------
    MemoryObject* Memory::createObject() const
    {
        return NULL;
    }
} 
