#include <memory.h>
#include <cuda_runtime.h>
#include <osgCuda/Context>
#include <osgCuda/Array>

namespace osgCuda
{
	/**
	*/
	class ArrayStream : public osgCompute::BufferStream
	{
	public:
		cudaArray*                      _devArray;
		bool                            _devArrayAllocated;
		bool                            _syncDevice;
		void*							_hostPtr;
		bool                            _hostPtrAllocated;
		bool                            _syncHost;
		unsigned int                    _allocHint;
		unsigned int                    _modifyCount;

		ArrayStream();
		virtual ~ArrayStream();

	private:
		// not allowed to call copy-constructor or copy-operator
		ArrayStream( const ArrayStream& ) {}
		ArrayStream& operator=( const ArrayStream& ) { return *this; }
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	ArrayStream::ArrayStream()
		:   osgCompute::BufferStream(),
			_devArray(NULL),
			_hostPtr(NULL),
			_syncDevice(false),
			_syncHost(false),
			_allocHint(0),
			_devArrayAllocated(false),
			_hostPtrAllocated(false),
			_modifyCount(UINT_MAX)
	{
	}

	//------------------------------------------------------------------------------
	ArrayStream::~ArrayStream()
	{
		if( _devArrayAllocated && NULL != _devArray )
		{
			cudaError_t res = cudaFreeArray( _devArray );
			if( res != cudaSuccess )
			{
				osg::notify(osg::FATAL)
					<<"ArrayStream::~ArrayStream(): error during cudaFreeArray(). "
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
	Array::Array()
		: osgCompute::Buffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void Array::clear()
	{
		clearLocal();
		osgCompute::Buffer::clear();
	}

	//------------------------------------------------------------------------------
	bool Array::init()
	{
		if( getNumDimensions() > 3 )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::init(): the maximum dimension allowed is 3."
				<< std::endl;

			clear();
			return false;
		}

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
					<< "osgCuda::Array::init(): image \""
					<< _image->getName() << "\" uses MipMaps which are currently"
					<< "not supported."
					<< std::endl;

				clear();
				return false;
			}

			if( _image->getTotalSizeInBytes() != byteSize )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::init(): size of image \""
					<< _image->getName() << "\" does not match the array size."
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
					<< "osgCuda::Array::init(): size of array \""
					<< _array->getName() << "\" is wrong."
					<< std::endl;

				clear();
				return false;
			}
		}

		return osgCompute::Buffer::init();
	}

	//------------------------------------------------------------------------------
	cudaArray* osgCuda::Array::mapArray( const osgCompute::Context& context, unsigned int mapping/*= osgCompute::MAP_DEVICE*/ ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::INFO)
				<< "osgCuda::Array::mapArray(): array is dirty."
				<< std::endl;

			return NULL;
		}

		//if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		//{
		//	osg::notify(osg::FATAL)
		//		<< "osgCuda::Array::mapArray(): calling thread differs from the context's thread."
		//		<< std::endl;

		//	return NULL;
		//}

		if( mapping & osgCompute::MAP_HOST )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::mapArray(): cannot map array to host. Call map() instead."
				<< std::endl;

			return NULL;
		}

		ArrayStream* stream = static_cast<ArrayStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::mapArray(): cannot receive ArrayStream for context \""
				<< context.getId() << "\"."
				<< std::endl;
			return NULL;
		}

		cudaArray* ptr = NULL;
		if( mapping != osgCompute::UNMAPPED )
			ptr = mapArrayStream( *stream, mapping );
		else
			unmapStream( *stream );

		return ptr;
	}

	//------------------------------------------------------------------------------
	void* Array::map( const osgCompute::Context& context, unsigned int mapping/*= osgCompute::MAP_DEVICE*/, unsigned int offset/*= 0*/, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::map(): array is dirty."
				<< std::endl;

			return NULL;
		}

		//if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		//{
		//	osg::notify(osg::FATAL)
		//		<< "osgCuda::Array::map(): calling thread differs from the context assigned thread."
		//		<< std::endl;

		//	return NULL;
		//}

		ArrayStream* stream = static_cast<ArrayStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::map(): cannot receive ArrayStream for context \""
				<< context.getId() << "\"."
				<< std::endl;
			return NULL;
		}

		if( mapping & osgCompute::MAP_DEVICE ) // lets try to avoid this in the future release
			return mapArrayStream( *stream, mapping );
		else if( mapping & osgCompute::MAP_HOST )
			return mapStream( *stream, mapping, offset );
		else
			unmapStream( *stream );

		return NULL;
	}

	//------------------------------------------------------------------------------
	void Array::unmap( const osgCompute::Context& context, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::unmap(): array is dirty."
				<< std::endl;

			return;
		}

		//if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		//{
		//	osg::notify(osg::FATAL)
		//		<< "osgCuda::Array::unmap(): calling thread differs from the context's thread."
		//		<< std::endl;

		//	return;
		//}

		ArrayStream* stream = static_cast<ArrayStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Array::unmap(): could not receive ArrayStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::Array::setMemory( const osgCompute::Context& context, int value, unsigned int mapping/*= osgCompute::MAP_DEVICE*/, unsigned int offset/*= 0*/, unsigned int count/*= UINT_MAX*/, unsigned int ) const
	{
		char* data = static_cast<char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::setMemory(): error during memset() for host within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}

			return true;
		}
		else if( mapping & osgCompute::MAP_DEVICE_TARGET )
		{
			osg::notify(osg::INFO)
				<< "osgCuda::Array::setMemory(): cudaMemset() for cuda arrays is not available yet."
				<< std::endl;

			return true;
		}

		unmap( context );
		return false;
	}

	//------------------------------------------------------------------------------
	cudaArray* Array::mapArrayStream( ArrayStream& stream, unsigned int mapping ) const
	{
		cudaArray* ptr = NULL;

		bool needsSetup = false;
		if( (_image.valid() && _image->getModifiedCount() != stream._modifyCount ) ||
			(_array.valid() && _array->getModifiedCount() != stream._modifyCount ) )
			needsSetup = true;

		///////////////////
		// PROOF MAPPING //
		///////////////////
		if( (stream._mapping & osgCompute::MAP_DEVICE && mapping & osgCompute::MAP_DEVICE) &&
			!needsSetup )
		{
			if( getSubloadCallback() && NULL != stream._devArray )
			{
				const osgCompute::SubloadCallback* callback = getSubloadCallback();
				if( callback )
				{
					// load or subload data before returning the host pointer
					callback->subload( reinterpret_cast<void*>(stream._devArray), mapping, 0, *this, *stream._context );
				}
			}

			stream._mapping = mapping;
			return stream._devArray;
		}
		else if( stream._mapping != osgCompute::UNMAPPED )
		{
			unmapStream( stream );
		}

		stream._mapping = mapping;

		//////////////
		// MAP DATA //
		//////////////
		bool firstLoad = false;
		if( (stream._mapping & osgCompute::MAP_DEVICE) )
		{
			if( NULL == stream._devArray )
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
			if( needsSetup )
				if( !setupStream( mapping, stream ) )
					return NULL;

			/////////////////
			// SYNC STREAM //
			/////////////////
			if( stream._syncDevice && NULL != stream._hostPtr )
				if( !syncStream( mapping, stream ) )
					return NULL;

			ptr = stream._devArray;
		}
		else
		{
			osg::notify(osg::WARN)
				<< "osgCuda::Array::mapArrayStream(): wrong mapping was specified. Use one of the following: "
				<< "DEVICE_SOURCE, DEVICE_TARGET, DEVICE."
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
					callback->load( reinterpret_cast<void*>(ptr), mapping, 0, *this, *stream._context );
				else
					callback->subload( reinterpret_cast<void*>(ptr), mapping, 0, *this, *stream._context );
			}
		}

		return ptr;
	}

	//------------------------------------------------------------------------------
	void* Array::mapStream( ArrayStream& stream, unsigned int mapping, unsigned int offset ) const
	{
		void* ptr = NULL;

		bool needsSetup = false;
		if( (_image.valid() && _image->getModifiedCount() != stream._modifyCount ) ||
			(_array.valid() && _array->getModifiedCount() != stream._modifyCount ) )
			needsSetup = true;

		///////////////////
		// PROOF MAPPING //
		///////////////////
		if( (stream._mapping & osgCompute::MAP_HOST && mapping & osgCompute::MAP_HOST) &&
			!needsSetup )
		{
			if( getSubloadCallback() && NULL != stream._hostPtr )
			{
				const osgCompute::SubloadCallback* callback = getSubloadCallback();
				if( callback )
				{
					// subload data before returning the pointer
					callback->subload( stream._hostPtr, mapping, offset, *this, *stream._context );
				}
			}

			stream._mapping = mapping;
			return &static_cast<char*>( stream._hostPtr )[offset];
		}
		else if( stream._mapping != osgCompute::UNMAPPED )
		{
			unmapStream( stream );
		}

		stream._mapping = mapping;

		//////////////
		// MAP DATA //
		//////////////
		bool firstLoad = false;
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
			if( needsSetup )
				if( !setupStream( mapping, stream ) )
					return NULL;

			/////////////////
			// SYNC STREAM //
			/////////////////
			if( stream._syncHost && NULL != stream._devArray )
				if( !syncStream( mapping, stream ) )
					return NULL;

			ptr = stream._hostPtr;
		}
		else
		{
			osg::notify(osg::WARN)
				<< "osgCuda::Array::mapStream(): wrong mapping specified. Use one of the following: "
				<< "HOST_SOURCE, HOST_TARGET, HOST."
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
	bool Array::setupStream( unsigned int mapping, ArrayStream& stream ) const
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
				data = reinterpret_cast<const void*>(_array->getDataPointer());
			}

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::setupStream(): Cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			if( getNumDimensions() < 3 )
			{
				res = cudaMemcpyToArray(stream._devArray,0,0,data, getByteSize(), cudaMemcpyHostToDevice);
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::setupStream(): cudaMemcpyToArray() failed for data within context \""
						<< stream._context->getId() << "\"." 
						<< " " << cudaGetErrorString( res ) << "."
						<< std::endl;

					return false;
				}
			}
			else
			{
				cudaMemcpy3DParms memCpyParams = {0};
				memCpyParams.dstArray = stream._devArray;
				memCpyParams.kind = cudaMemcpyHostToDevice;
				memCpyParams.srcPtr = make_cudaPitchedPtr((void*)data, getDimension(0)*getElementSize(), getDimension(0), getDimension(1));

				cudaExtent arrayExtent = {0};
				arrayExtent.width = getDimension(0);
				arrayExtent.height = getDimension(1);
				arrayExtent.depth = getDimension(2);

				memCpyParams.extent = arrayExtent;

				res = cudaMemcpy3D( &memCpyParams );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::setupStream(): cudaMemcpy3D() failed for data within context \""
						<< stream._context->getId() << "\"." 
						<< " " << cudaGetErrorString( res ) <<"."
						<< std::endl;

					return false;
				}
			}


			// host must be synchronized
			stream._syncHost = true;
			stream._modifyCount = _image.valid()? _image->getModifiedCount() : _array->getModifiedCount();
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
					<< "osgCuda::Array::setupStream(): cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._hostPtr, data, getByteSize(), cudaMemcpyHostToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::setupStream(): error during cudaMemcpy() within context \""
					<< stream._context->getId() << "\"." 
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			// device must be synchronized
			stream._syncDevice = true;
			stream._modifyCount = _image.valid()? _image->getModifiedCount() : _array->getModifiedCount();
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Array::allocStream( unsigned int mapping, ArrayStream& stream ) const
	{
		if( mapping & osgCompute::MAP_HOST )
		{
			if( stream._hostPtr != NULL )
				return true;

			stream._hostPtr = malloc( getByteSize() );
			if( NULL == stream._hostPtr )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::allocStream(): something goes wrong within mallocHostMemory() within Context \""<<stream._context->getId()
					<< "\"."
					<< std::endl;

				return false;
			}

			stream._hostPtrAllocated = true;
			if( stream._devArray != NULL )
				stream._syncHost = true;
			return true;
		}
		else if( mapping & osgCompute::MAP_DEVICE )
		{
			if( stream._devArray != NULL )
				return true;

			const cudaChannelFormatDesc& desc = getChannelFormatDesc();
			if( desc.x == INT_MAX && desc.y == INT_MAX && desc.z == INT_MAX )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::allocStream(): no valid ChannelFormatDesc found."
					<< std::endl;

				return false;
			}

			if( getNumDimensions() == 3 )
			{

				cudaExtent extent;
				extent.width = getDimension(0);
				extent.height = (getDimension(1) <= 1)? 0 : getDimension(1);
				extent.depth = (getDimension(2) <= 1)? 0 : getDimension(2);

				// allocate memory
				cudaError_t res = cudaMalloc3DArray( &stream._devArray, &desc, extent );
				if( cudaSuccess != res || NULL ==  stream._devArray )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::allocStream(): something goes wrong within mallocDevice3DArray() within context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}
			}
			else if( getNumDimensions() == 2 )
			{
				cudaError_t res = cudaMallocArray( &stream._devArray, &desc, getDimension(0), (getDimension(1) <= 1)? 0 : getDimension(1) );
				if( cudaSuccess != res || NULL ==  stream._devArray )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::allocStream(): something goes wrong within mallocDevice2DArray() within context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}
			}
			else 
			{
				cudaError_t res = cudaMallocArray( &stream._devArray, &desc, getDimension(0), 1 );
				if( cudaSuccess != res || NULL ==  stream._devArray )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::allocStream(): something goes wrong within mallocDeviceArray() within context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}
			}

			stream._devArrayAllocated = true;
			if( stream._hostPtr != NULL )
				stream._syncDevice = true;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Array::syncStream( unsigned int mapping, ArrayStream& stream ) const
	{
		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			if( getNumDimensions() == 1 )
			{
				res = cudaMemcpyToArray( stream._devArray, 0, 0, stream._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::syncStream(): error during cudaMemcpyToArray() to device within context \""
						<< stream._context->getId() << "\"."
						<< " " << cudaGetErrorString( res )<<"."
						<< std::endl;
					return false;
				}
			}
			else if( getNumDimensions() == 2 )
			{
				res = cudaMemcpy2DToArray( stream._devArray,
					0, 0,
					stream._hostPtr,
					getDimension(0) * getElementSize(),
					getDimension(0),
					getDimension(1),
					cudaMemcpyHostToDevice );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::syncStream(): error during cudaMemcpy2DToArray() to device within context \""
						<< stream._context->getId() << "\"."
						<< " " << cudaGetErrorString( res ) << "."
						<< std::endl;

					return false;
				}
			}
			else
			{
				cudaPitchedPtr pitchPtr = {0};
				pitchPtr.pitch = getDimension(0) * getElementSize();
				pitchPtr.ptr = (void*)stream._hostPtr;
				pitchPtr.xsize = getDimension(0);
				pitchPtr.ysize = getDimension(1);

				cudaExtent extent = {0};
				extent.width = getDimension(0);
				extent.height = getDimension(1);
				extent.depth = getDimension(2);

				cudaMemcpy3DParms copyParams = {0};
				copyParams.srcPtr = pitchPtr;
				copyParams.dstArray = stream._devArray;
				copyParams.extent = extent;
				copyParams.kind = cudaMemcpyHostToDevice;

				res = cudaMemcpy3D( &copyParams );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::syncStream(): error during cudaMemcpy3D() to device within context \""
						<< stream._context->getId() << "\"."
						<< " " << cudaGetErrorString( res ) << "."
						<< std::endl;

					return false;
				}
			}

			stream._syncDevice = false;
			return true;
		}
		else if( mapping & osgCompute::MAP_HOST )
		{
			if( getNumDimensions() == 1 )
			{
				res = cudaMemcpyFromArray( stream._hostPtr, stream._devArray, 0, 0, getByteSize(), cudaMemcpyDeviceToHost );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::syncStream(): something goes wrong within cudaMemcpyFromArray() to host within context \""
						<< stream._context->getId() << "\"."
						<< " " << cudaGetErrorString( res ) <<"."
						<< std::endl;

					return false;
				}
			}
			else if( getNumDimensions() == 2 )
			{
				res = cudaMemcpy2DFromArray(
					stream._hostPtr,
					getDimension(0) * getElementSize(),
					stream._devArray,
					0, 0,
					getDimension(0),
					getDimension(1),
					cudaMemcpyDeviceToHost );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::syncStream(): error during cudaMemcpy2DFromArray() to device within context \""
						<< stream._context->getId() << "\"."
						<< " " << cudaGetErrorString( res ) <<"."
						<< std::endl;

					return false;
				}
			}
			else
			{
				cudaPitchedPtr pitchPtr = {0};
				pitchPtr.pitch = getDimension(0)*getElementSize();
				pitchPtr.ptr = (void*)stream._hostPtr;
				pitchPtr.xsize = getDimension(0);
				pitchPtr.ysize = getDimension(1);

				cudaExtent extent = {0};
				extent.width = getDimension(0);
				extent.height = getDimension(1);
				extent.depth = getDimension(2);

				cudaMemcpy3DParms copyParams = {0};
				copyParams.srcArray = stream._devArray;
				copyParams.dstPtr = pitchPtr;
				copyParams.extent = extent;
				copyParams.kind = cudaMemcpyDeviceToHost;

				res = cudaMemcpy3D( &copyParams );
				if( cudaSuccess != res )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Array::syncStream(): error during cudaMemcpy3D() to device within context \""
						<< stream._context->getId() << "\"."
						<< " " << cudaGetErrorString( res ) <<"."
						<< std::endl;

					return false;

				}
			}

			stream._syncHost = false;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	void Array::unmapStream( ArrayStream& stream ) const
	{
		if( (stream._mapping & osgCompute::MAP_HOST_TARGET) )
		{
			stream._syncDevice = true;
		}
		else if( (stream._mapping & osgCompute::MAP_DEVICE_TARGET) )
		{
			stream._syncHost = true;
		}

		stream._mapping = osgCompute::UNMAPPED;
	}

	//------------------------------------------------------------------------------
	void Array::setImage( osg::Image* image )
	{
		if( !osgCompute::Resource::isClear() && NULL != image)
		{
			if( image->getNumMipmapLevels() > 1 )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::setupStream(): image \""
					<< image->getName() << "\" uses MipMaps which are currently"
					<< "not supported."
					<< std::endl;

				return;
			}

			if( image->getTotalSizeInBytes() != getByteSize() )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::setupStream(): size of image \""
					<< image->getName() << "\" is wrong."
					<< std::endl;

				return;
			}
		}

		_image = image;
		_array = NULL;
	}

	//------------------------------------------------------------------------------
	osg::Image* Array::getImage()
	{
		return _image.get();
	}

	//------------------------------------------------------------------------------
	const osg::Image* Array::getImage() const
	{
		return _image.get();
	}

	//------------------------------------------------------------------------------
	void Array::setArray( osg::Array* array )
	{
		if( !osgCompute::Resource::isClear() && array != NULL )
		{
			if( array->getTotalDataSize() != getByteSize() )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Array::setArray(): size of array \""
					<< array->getName() << "\" does not match with the array size."
					<< std::endl;

				return;
			}
		}

		_array = array;
		_image = NULL;
	}

	//------------------------------------------------------------------------------
	osg::Array* Array::getArray()
	{
		return _array.get();
	}

	//------------------------------------------------------------------------------
	const osg::Array* Array::getArray() const
	{
		return _array.get();
	}


	//------------------------------------------------------------------------------
	cudaChannelFormatDesc& Array::getChannelFormatDesc()
	{
		return _channelFormatDesc;
	}

	//------------------------------------------------------------------------------
	const cudaChannelFormatDesc& Array::getChannelFormatDesc() const
	{
		return _channelFormatDesc;
	}

	//------------------------------------------------------------------------------
	void Array::setChannelFormatDesc(cudaChannelFormatDesc& channelFormatDesc)
	{
		if( !osgCompute::Resource::isClear() )
			return;

		_channelFormatDesc = channelFormatDesc;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Array::clearLocal()
	{
		_image = NULL;
		_array = NULL;
		memset( &_channelFormatDesc, INT_MAX, sizeof(cudaChannelFormatDesc) );
	}

	//------------------------------------------------------------------------------
	osgCompute::BufferStream* Array::newStream( const osgCompute::Context& context ) const
	{
		return new ArrayStream;
	}

}