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
#include <osgCompute/Buffer>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    BufferStream::BufferStream() 
        :   _mapping( UNMAPPED ),
            _allocHint(0)
    {
    }

    //------------------------------------------------------------------------------
    BufferStream::~BufferStream() 
    {
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Buffer::Buffer() 
        : Resource()
    { 
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Buffer::clear()
    {
        Resource::clear();
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool Buffer::init()
    {
        if( !isClear() )
            return true;

        if( _dimensions.empty() )
        {
            osg::notify(osg::FATAL)  
                << "Buffer::init(): no dimensions specified."                  
                << std::endl;

            return false;
        }

		if( _elementSize == 0 )
		{
			osg::notify(osg::FATAL)  
				<< "Buffer::init(): no element size specified."                  
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
	void Buffer::setElementSize( unsigned int elementSize ) 
	{ 
		if( !isClear() )
			return;

		_elementSize = elementSize; 
	}

	//------------------------------------------------------------------------------
	unsigned int Buffer::getElementSize() const 
	{ 
		return _elementSize; 
	}

	//------------------------------------------------------------------------------
	unsigned int Buffer::getByteSize() const 
	{ 
		return getElementSize() * getNumElements(); 
	}

	//------------------------------------------------------------------------------
	void Buffer::setDimension( unsigned int dimIdx, unsigned int dimSize )
	{
		if( !isClear() )
			return;

		if (_dimensions.size()<=dimIdx)
			_dimensions.resize(dimIdx+1,0);

		_dimensions[dimIdx] = dimSize;
	}

	//------------------------------------------------------------------------------
	unsigned int Buffer::getDimension( unsigned int dimIdx ) const
	{ 
		if( dimIdx > (_dimensions.size()-1) )
			return 0;

		return _dimensions[dimIdx];
	}

	//------------------------------------------------------------------------------
	unsigned int Buffer::getNumDimensions() const
	{ 
		return _dimensions.size();
	}

	//------------------------------------------------------------------------------
	unsigned int osgCompute::Buffer::getNumElements() const
	{
		return _numElements;
	}

	//------------------------------------------------------------------------------
	void osgCompute::Buffer::setAllocHint( unsigned int allocHint )
	{
		if( !isClear() )
			return;

		_allocHint = (_allocHint | allocHint);
	}

	//------------------------------------------------------------------------------
	unsigned int osgCompute::Buffer::getAllocHint() const
	{
		return _allocHint;
	}

	//------------------------------------------------------------------------------
	void Buffer::setSubloadCallback( SubloadCallback* sc ) 
	{ 
		_subloadCallback = sc; 
	}

	//------------------------------------------------------------------------------
	SubloadCallback* Buffer::getSubloadCallback() 
	{ 
		return _subloadCallback.get(); 
	}

	//------------------------------------------------------------------------------
	const SubloadCallback* Buffer::getSubloadCallback() const 
	{ 
		return _subloadCallback.get(); 
	}

    //------------------------------------------------------------------------------
    unsigned int Buffer::getMapping( const Context& context, unsigned int ) const
    {
        if( isClear() )
            return osgCompute::UNMAPPED;

        BufferStream* stream = lookupStream( context );
        if( NULL == stream )
        {
            osg::notify(osg::FATAL)  
                << "Buffer::getMapping(): cannot receive stream for \""
                << context.getId() << "\"."
                << std::endl;

            return osgCompute::UNMAPPED;
        }

        return stream->_mapping;
    }

	//------------------------------------------------------------------------------
	void Buffer::swap( unsigned int incr /*= 1 */ )
	{
		// Function should be implemented by swap buffers.
	}

	//------------------------------------------------------------------------------
	unsigned int Buffer::getSwapCount() const
	{
		// Function should be implemented by swap buffers.
		return 0;
	}

	//------------------------------------------------------------------------------
	void Buffer::init( const Context& context ) const
	{
		if( _streams.size()<= context.getId() )
			_streams.resize( context.getId()+1, NULL );

		if( NULL == _streams[context.getId()] )
		{
			// create new buffer and attach it to the
			// current context
			_streams[context.getId()] = newStream( context );

			if( NULL == _streams[context.getId()] )
			{
				osg::notify( osg::FATAL )  
					<< "Buffer::lookupStream(): allocation of data stream failed for context \""<<context.getId()<<"\"."
					<< std::endl;

				return;
			}
			_streams[context.getId()]->_allocHint = getAllocHint();
			_streams[context.getId()]->_mapping = UNMAPPED;
			_streams[context.getId()]->_context = const_cast<osgCompute::Context*>( &context );
			Resource::init( context );
		}
	}

	//------------------------------------------------------------------------------
	void Buffer::clear( const Context& context ) const
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

		BufferStream* curStream = _streams[context.getId()];
		if( curStream != NULL )
		{
			std::vector<BufferStream*>::iterator itr = _streams.begin();
			while( (*itr) != curStream )
				itr++;

			// delete stream and detach 
			// it this from context
			delete curStream;
			_streams.erase( itr );
		}

		Resource::clear( context );
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Buffer::~Buffer() 
	{ 
		clearLocal(); 
	}

	//------------------------------------------------------------------------------
	void Buffer::clearLocal()
	{
		_dimensions.clear();
		_numElements = 0;
		_elementSize = 0;
		_allocHint = NO_ALLOC_HINT;
		_subloadCallback = NULL;

		for( unsigned int s=0; s<_streams.size(); ++s )
			if( _streams[s] != NULL )
				delete _streams[s];
		_streams.clear();
	}

	//------------------------------------------------------------------------------
	BufferStream* Buffer::lookupStream( const Context& context ) const
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

		/////////////////////
		// ALLOCATE STREAM //
		/////////////////////
		if( _streams.size() <= context.getId() || 
			NULL == _streams[context.getId()] )
			init(context);

		return _streams[context.getId()];
	}

	//------------------------------------------------------------------------------
	BufferStream* Buffer::newStream( const Context& ) const
	{
		// implemented in the sub-classes
		return NULL;
	}
}
