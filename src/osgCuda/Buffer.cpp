#include <memory.h>
#if defined(__linux)
    #include <malloc.h>
#endif
#include <osgCuda/Context>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <osgCuda/Buffer>

namespace osgCuda
{
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	BufferStream::BufferStream()
		:   osgCompute::BufferStream(),
			_devPtr(NULL),
			_hostPtr(NULL),
			_syncDevice(false),
			_syncHost(false),
			_devPtrAllocated(false),
			_hostPtrAllocated(false),
			_modifyCount(UINT_MAX)
	{
	}

	//------------------------------------------------------------------------------
	BufferStream::~BufferStream()
	{
		if( _devPtrAllocated && NULL != _devPtr)
		{
			cudaError_t res = cudaFree( _devPtr );
			if( res != cudaSuccess )
			{
				osg::notify(osg::FATAL)
					<<"BufferStream::~BufferStream(): error during cudaFreeArray(). "
					<<cudaGetErrorString(res)<<std::endl;
			}
		}


		if( _hostPtrAllocated && NULL != _hostPtr)
			free( _hostPtr );
	}



	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Buffer::Buffer()
		: osgCompute::Buffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void Buffer::clear()
	{
		clearLocal();
		osgCompute::Buffer::clear();
	}

	//------------------------------------------------------------------------------
	bool Buffer::init()
	{
		unsigned int numElements = 1;
		for( unsigned int d=0; d<getNumDimensions(); ++d )
			numElements *= getDimension( d );

		unsigned int byteSize = numElements * getElementSize();

		// check stream data
		if( _image.valid() )
		{
			if( _image->getNumMipmapLevels() > 1 )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::init(): Image \""
					<< _image->getName() << "\" uses MipMaps which are currently"
					<< "not supported."
					<< std::endl;

				clear();
				return false;
			}

			if( _image->getTotalSizeInBytes() != byteSize )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::init(): size of image \""
					<< _image->getName() << "\" is wrong."
					<< std::endl;

				clear();
				return false;
			}
		}

		if( _array.valid() )
		{
			if( _array->getTotalDataSize() != byteSize )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::init(): size of array \""
					<< _array->getName() << "\" is wrong."
					<< std::endl;

				clear();
				return false;
			}
		}

		return osgCompute::Buffer::init();
	}

	//------------------------------------------------------------------------------
	void* Buffer::map( const osgCompute::Context& context, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map(): buffer is dirty."
				<< std::endl;

			return NULL;
		}

		if( mapping == osgCompute::UNMAPPED )
		{
			unmap( context, hint );
			return NULL;
		}

		BufferStream* stream = static_cast<BufferStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map(): could not receive BufferStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return NULL;
		}

		void* ptr = mapStream( *stream, mapping, offset );
		if(NULL !=  ptr )
		{
			if( (mapping & osgCompute::MAP_DEVICE_TARGET) == osgCompute::MAP_DEVICE_TARGET )
				stream->_syncHost = true;

			if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
				stream->_syncDevice = true;
		}
		else unmapStream( *stream );

		return ptr;
	}

	//------------------------------------------------------------------------------
	void Buffer::unmap( const osgCompute::Context& context, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map(): buffer is dirty."
				<< std::endl;

			return;
		}

		BufferStream* stream = static_cast<BufferStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map(): could not receive BufferStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::Buffer::setMemory( const osgCompute::Context& context, int value, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int count/* = UINT_MAX*/, unsigned int ) const
	{
		unsigned char* data = static_cast<unsigned char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setMemory(): error during memset() for host within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}
		}
		else if( mapping & osgCompute::MAP_DEVICE_TARGET )
		{
			cudaError res = cudaMemset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count );
			if( res != cudaSuccess )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setMemory(): error during cudaMemset() for device data within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}
		}

		return true;
	}

	//------------------------------------------------------------------------------
	bool Buffer::resetMemory( const osgCompute::Context& context, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::resetMemory(): buffer is dirty."
				<< std::endl;

			return false;
		}

		BufferStream* stream = static_cast<BufferStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::resetMemory(): could not receive BufferStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return false;
		}

		// reset array or image data
		stream->_modifyCount = UINT_MAX;

		// clear host memory
		if( stream->_hostPtr != NULL )
		{
			if( !memset( stream->_hostPtr, 0x0, getByteSize() ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::resetMemory(): error during memset() for host data within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}


			stream->_mapping = osgCompute::MAP_HOST;
			stream->_syncHost = false;
		}

		// clear device memory
		if( stream->_devPtr != NULL )
		{
			cudaError res = cudaMemset( stream->_devPtr, 0x0, getByteSize() );
			if( res != cudaSuccess )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::resetMemory(): error during cudaMemset() for device data within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}

			stream->_mapping = osgCompute::MAP_DEVICE;
			stream->_syncDevice = false;
		}

		return true;
	}

	//------------------------------------------------------------------------------
	void* Buffer::mapStream( BufferStream& stream, unsigned int mapping, unsigned int offset ) const
	{
		void* ptr = NULL;

		bool firstLoad = false;
		// check for modifications
		bool needsSetup = false;
		if( (_image.valid() && _image->getModifiedCount() != stream._modifyCount ) ||
			(_array.valid() && _array->getModifiedCount() != stream._modifyCount ) )
			needsSetup = true;

		// current mapping
		stream._mapping = mapping;

		//////////////
		// MAP DATA //
		//////////////
		if( (stream._mapping & osgCompute::MAP_HOST) )
		{
			if( NULL == stream._hostPtr )
			{
				//////////////////////////
				// ALLOCATE HOST-MEMORY //
				//////////////////////////
				if( !allocStream( mapping, stream ) )
					return NULL;

				firstLoad = true;
			}

			//////////////////
			// SETUP STREAM //
			//////////////////
			if( needsSetup && !stream._syncHost )
				if( !setupStream( mapping, stream ) )
					return NULL;

			/////////////////
			// SYNC STREAM //
			/////////////////
			if( stream._syncHost && NULL != stream._devPtr )
				if( !syncStream( mapping, stream ) )
					return NULL;

			ptr = stream._hostPtr;
		}
		else if( (stream._mapping & osgCompute::MAP_DEVICE) )
		{
			if( NULL == stream._devPtr )
			{
				////////////////////////////
				// ALLOCATE DEVICE-MEMORY //
				////////////////////////////
				if( !allocStream( mapping, stream ) )
					return NULL;

				firstLoad = true;
			}

			//////////////////
			// SETUP STREAM //
			//////////////////
			if( needsSetup && !stream._syncDevice )
				if( !setupStream( mapping, stream ) )
					return NULL;

			/////////////////
			// SYNC STREAM //
			/////////////////
			if( stream._syncDevice && NULL != stream._hostPtr )
				if( !syncStream( mapping, stream ) )
					return NULL;

			ptr = stream._devPtr;
		}
		else
		{
			osg::notify(osg::WARN)
				<< "osgCuda::Buffer::mapStream(): Wrong mapping. Use one of the following: "
				<< "HOST_SOURCE, HOST_TARGET, HOST, DEVICE_SOURCE, DEVICE_TARGET, DEVICE."
				<< std::endl;

			return NULL;
		}

		//////////////////
		// LOAD/SUBLOAD //
		//////////////////
		if( getSubloadCallback() && NULL != ptr )
		{
			const osgCompute::SubloadCallback* callback = getSubloadCallback();
			if( callback )
			{
				// load or subload data before returning the host pointer
				if( firstLoad )
					callback->load( ptr, mapping, offset, *this, *stream._context );
				else
					callback->subload( ptr, mapping, offset, *this, *stream._context );
			}
		}

		return &static_cast<char*>(ptr)[offset];
	}

	//------------------------------------------------------------------------------
	bool Buffer::setupStream( unsigned int mapping, BufferStream& stream ) const
	{
		cudaError res;

		if( mapping & osgCompute::MAP_DEVICE )
		{
			const void* data = NULL;
			if( _image.valid() )
			{
				data = _image->data();
			}

			if( _array.valid() )
			{
				data = reinterpret_cast<const void*>( _array->getDataPointer() );
			}

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream(): cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._devPtr,  data, getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream(): error during cudaMemcpy() within context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			// host must be synchronized
			// because device memory has been modified
			stream._syncHost = true;
			stream._modifyCount = _image.valid()? _image->getModifiedCount() : UINT_MAX;
			stream._modifyCount = _array.valid()? _array->getModifiedCount() : UINT_MAX;

			return true;
		}
		else if( mapping & osgCompute::MAP_HOST )
		{
			const void* data = NULL;
			if( _image.valid() )
			{
				data = _image->data();
			}

			if( _array.valid() )
			{
				data = reinterpret_cast<const void*>( _array->getDataPointer() );
			}

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream(): cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._hostPtr,  data, getByteSize(), cudaMemcpyHostToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream(): error during cudaMemcpy() within context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			// device must be synchronized
			// because host memory has been modified
			stream._syncDevice = true;
			stream._modifyCount = _image.valid()? _image->getModifiedCount() : UINT_MAX;
			stream._modifyCount = _array.valid()? _array->getModifiedCount() : UINT_MAX;

			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Buffer::allocStream( unsigned int mapping, BufferStream& stream ) const
	{
		if( mapping & osgCompute::MAP_HOST )
		{
			if( stream._hostPtr != NULL )
				return true;

			stream._hostPtr = malloc( getByteSize() );
			if( NULL == stream._hostPtr )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::allocStream(): error during mallocDeviceHost() within Context \""<<stream._context->getId()
					<< "\"."
					<< std::endl;

				return false;
			}

			// clear memory
			memset( stream._hostPtr, 0x0, getByteSize() );

			stream._hostPtrAllocated = true;
			if( stream._devPtr != NULL )
				stream._syncHost = true;
			return true;
		}
		else if( mapping & osgCompute::MAP_DEVICE )
		{
			if( stream._devPtr != NULL )
				return true;

			if( getNumDimensions() == 3 )
			{
				cudaPitchedPtr pitchPtr;
				cudaExtent extent;
				extent.width = getDimension(0) * getElementSize();
				extent.height = getDimension(1);
				extent.depth = getDimension(2);

				// allocate memory
				cudaError_t res = cudaMalloc3D( &pitchPtr, extent );
				if( cudaSuccess != res || NULL == pitchPtr.ptr )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream(): error during mallocDevice3D() within context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}

				stream._devPtr = pitchPtr.ptr;

				// clear memory
				cudaMemset3D( pitchPtr, 0x0, extent );
			}
			else if( getNumDimensions() == 2 )
			{
				size_t pitch = 0;
				cudaError_t res = cudaMallocPitch( &stream._devPtr, &pitch, getDimension(0) * getElementSize(), getDimension(1) );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream(): error during mallocDevice2D() within context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}


				// clear memory
				cudaMemset2D( stream._devPtr, pitch, 0x0, getDimension(0) * getElementSize(), getDimension(1) );
			}
			else
			{
				cudaError_t res = cudaMalloc( &stream._devPtr, getByteSize() );
				if( res != cudaSuccess )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream(): error during mallocDevice() within context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}

				// clear memory
				cudaMemset( stream._devPtr, 0x0, getByteSize() );
			}


			stream._devPtrAllocated = true;

			if( stream._hostPtr != NULL )
				stream._syncDevice = true;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Buffer::syncStream( unsigned int mapping, BufferStream& stream ) const
	{
		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			res = cudaMemcpy( stream._devPtr, stream._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::syncStream(): error during cudaMemcpy() to device within Context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;
				return false;
			}

			stream._syncDevice = false;
			return true;
		}
		else if( mapping & osgCompute::MAP_HOST )
		{
			res = cudaMemcpy( stream._hostPtr, stream._devPtr, getByteSize(), cudaMemcpyDeviceToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::syncStream(): error during cudaMemcpy() to host within Context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			stream._syncHost = false;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	void Buffer::unmapStream( BufferStream& stream ) const
	{
		stream._mapping = osgCompute::UNMAPPED;
	}

	//------------------------------------------------------------------------------
	void Buffer::setImage( osg::Image* image )
	{
		if( !osgCompute::Resource::isClear() && image != NULL )
		{
			if( image->getNumMipmapLevels() > 1 )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream(): image \""
					<< image->getName() << "\" uses MipMaps which are currently"
					<< "not supported."
					<< std::endl;

				return;
			}

			if( image->getTotalSizeInBytes() != getByteSize() )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream(): size of image \""
					<< image->getName() << "\" is wrong."
					<< std::endl;

				return;
			}
		}

		_image = image;
		_array = NULL;
		resetModifiedCounts();
	}

	//------------------------------------------------------------------------------
	osg::Image* Buffer::getImage()
	{
		return _image.get();
	}

	//------------------------------------------------------------------------------
	const osg::Image* Buffer::getImage() const
	{
		return _image.get();
	}

	//------------------------------------------------------------------------------
	void Buffer::setArray( osg::Array* array )
	{
		if( !osgCompute::Resource::isClear() && array != NULL )
		{
			if( array->getTotalDataSize() != getByteSize() )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setArray(): size of array \""
					<< array->getName() << "\" is wrong."
					<< std::endl;

				return;
			}
		}

		_array = array;
		_image = NULL;
		resetModifiedCounts();
	}

	//------------------------------------------------------------------------------
	osg::Array* Buffer::getArray()
	{
		return _array.get();
	}

	//------------------------------------------------------------------------------
	const osg::Array* Buffer::getArray() const
	{
		return _array.get();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Buffer::clearLocal()
	{
		_array = NULL;
		_image = NULL;
	}

	//------------------------------------------------------------------------------
	osgCompute::BufferStream* Buffer::newStream( const osgCompute::Context& context ) const
	{
		return new BufferStream;
	}

	//------------------------------------------------------------------------------
	void Buffer::resetModifiedCounts() const
	{
		for( std::vector<osgCompute::BufferStream*>::iterator itr = osgCompute::Buffer::_streams.begin(); 
			 itr != osgCompute::Buffer::_streams.end();
			 ++itr )
		{
			osgCuda::BufferStream* curStream = dynamic_cast< osgCuda::BufferStream* >( (*itr) );
			curStream->_modifyCount = UINT_MAX;
		}
	}
}
