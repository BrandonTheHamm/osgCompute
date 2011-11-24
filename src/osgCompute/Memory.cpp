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
#include <osg/RenderInfo>
#include <osgCompute/Memory>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    MemoryObject::MemoryObject() 
        :   Referenced(),
            _mapping( UNMAP ),
			_allocHint(0),
			_syncOp(NO_SYNC),
            _pitch(0)
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
        _dimensions.clear();
        _numElements = 0;
        _elementSize = 0;
        _allocHint = 0;
        _subloadCallback = NULL;
        _pitch = 0;
    }

    //------------------------------------------------------------------------------
    void Memory::setElementSize( unsigned int elementSize ) 
    { 
        if( _object.valid() ) 
            releaseObjects();

        _elementSize = elementSize; 
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getElementSize() const 
    { 
        return _elementSize; 
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getAllElementsSize( unsigned int hint /*= 0 */ ) const 
    { 
        return getElementSize() * getNumElements(); 
    }
    

    //------------------------------------------------------------------------------
    unsigned int Memory::getByteSize( unsigned int mapping, unsigned int hint /*= 0 */ ) const 
    { 
        return 0;
    }

    //------------------------------------------------------------------------------
    void Memory::setDimension( unsigned int dimIdx, unsigned int dimSize )
    {
        if( _object.valid() ) 
            releaseObjects();

        if (_dimensions.size()<=dimIdx)
            _dimensions.resize(dimIdx+1,0);

        _dimensions[dimIdx] = dimSize;

		_numElements = 1;
		for( unsigned int d=0; d<_dimensions.size(); ++d )
			_numElements *= _dimensions[d];
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getDimension( unsigned int dimIdx ) const
    { 
        if( _dimensions.empty() || dimIdx > (_dimensions.size()-1) )
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
        if( _object.valid() ) 
            releaseObjects();

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
        if( _object.valid() )
            return _object->_mapping;
        else 
            return osgCompute::UNMAP;
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getPitch( unsigned int hint /*= 0 */ ) const
    {
        if( !_object.valid() )
            return computePitch();

        if( _pitch == 0 )
            _pitch = computePitch();
                
        return _pitch;
    }


    //------------------------------------------------------------------------------
    void Memory::swap( unsigned int )
    {
        // Function should be implemented by swap buffers.
    }

    //------------------------------------------------------------------------------
    void Memory::setSwapCount( unsigned int )
    {
        // Function should be implemented by swap buffers.
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getSwapCount() const
    {
        // Function should be implemented by swap buffers.
        return 1;
    }

    //------------------------------------------------------------------------------
    void Memory::setSwapIdx( unsigned int )
    {
        // Function should be implemented by swap buffers.
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getSwapIdx() const
    {
        // Function should be implemented by swap buffers.
        return 0;
    }

    //------------------------------------------------------------------------------
    void Memory::clear()
    {
        _dimensions.clear();
        _allocHint = 0x0;
        _elementSize = 0;
        _pitch = 0;
        _numElements = 0;
        Resource::clear();
    }

    //------------------------------------------------------------------------------
    void Memory::releaseObjects()
    {
        _object = NULL;
        Resource::releaseObjects();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Memory::~Memory() 
    { 
    }

    //------------------------------------------------------------------------------
    MemoryObject* Memory::object( bool create /*= true*/ )
    {
        if( !_object.valid() && create)
        {
            if( getElementSize() == 0 || getNumDimensions() == 0 )
            {
                osg::notify( osg::FATAL )  
                    << getName() << " [Memory::object()] \"" << getName() 
                    << "\": allocation of memory failed as dimension and element size is unknown."
                    << std::endl;

                return NULL;
            }

            MemoryObject* newObject = createObject();
            if( newObject == NULL )
            {
                osg::notify( osg::FATAL )  
                    << getName() << " [Memory::getObject()] \"" << getName() << "\": allocation of memory failed."
                    << std::endl;
                return NULL;
            }
            newObject->_mapping = osgCompute::UNMAP;
            newObject->_allocHint = getAllocHint();
            _object = newObject;
            objectsCreated();
        }

        return _object.get();
    }

    //------------------------------------------------------------------------------
    const MemoryObject* Memory::object( bool create /*= true*/ ) const
    {
        if( !_object.valid() && create)
        {
            if( getElementSize() == 0 || getNumDimensions() == 0 )
            {
                osg::notify( osg::FATAL )  
                    << getName() << " [Memory::object()] \"" << getName() 
                    << "\": allocation of memory failed as dimension and element size is unknown."
                    << std::endl;

                return NULL;
            }

            MemoryObject* newObject = createObject();
            if( newObject == NULL )
            {
                osg::notify( osg::FATAL )  
                    << getName() << " [Memory::object()] \"" << getName() << "\": allocation of memory failed."
                    << std::endl;

                return NULL;
            }
            newObject->_mapping = osgCompute::UNMAP;
            newObject->_allocHint = getAllocHint();
            _object = newObject;
            objectsCreated();
        }

        return _object.get();
    }

    //------------------------------------------------------------------------------
    MemoryObject* Memory::createObject() const
    {
        return NULL;
    }

    //------------------------------------------------------------------------------
    unsigned int Memory::getAllocatedByteSize( unsigned int mapping, unsigned int hint /*= 0 */ ) const
    {
        return 0;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
	// STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
    osg::observer_ptr<osg::GraphicsContext> GLMemory::s_context = NULL;

	//------------------------------------------------------------------------------
	void GLMemory::bindToContext( osg::GraphicsContext& context )
	{
		s_context = &context;
	}

	//------------------------------------------------------------------------------
	void GLMemory::releaseContext()
	{
		s_context = NULL;
	}

	//------------------------------------------------------------------------------
    osg::GraphicsContext* GLMemory::getContext()
	{
		return s_context.get();
	}
} 
