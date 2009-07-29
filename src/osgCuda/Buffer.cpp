#include <memory.h>
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
			static_cast<Context*>(osgCompute::BufferStream::_context.get())->freeMemory( _devPtr );
		if( _hostPtrAllocated && NULL != _hostPtr)
			static_cast<Context*>(osgCompute::BufferStream::_context.get())->freeMemory( _hostPtr );
	}



	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Buffer::Buffer()
		: osgCompute::Buffer(),
		  osg::Object()
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
		for( unsigned int d=0; d<osgCompute::Buffer::getNumDimensions(); ++d )
			numElements *= osgCompute::Buffer::getDimension( d );

		unsigned int byteSize = numElements * getElementSize();

		// check stream data
		if( _image.valid() )
		{
			if( _image->getNumMipmapLevels() > 1 )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::init() for Buffer \""
					<< osg::Object::getName() <<"\": Image \""
					<< _image->getName() << "\" uses MipMaps which are currently"
					<< "not supported."
					<< std::endl;

				clear();
				return false;
			}

			if( _image->getTotalSizeInBytes() != byteSize )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::init() for buffer \""
					<< osg::Object::getName() <<"\": size of image \""
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
					<< "osgCuda::Buffer::init() for buffer \""
					<< osg::Object::getName() <<"\": size of array \""
					<< _array->getName() << "\" is wrong."
					<< std::endl;

				clear();
				return false;
			}
		}

		return osgCompute::Buffer::init();
	}

	//------------------------------------------------------------------------------
	void* Buffer::map( const osgCompute::Context& context, unsigned int mapping ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map() for buffer \""
				<< osg::Object::getName() <<"\": buffer is dirty."
				<< std::endl;

			return NULL;
		}

		if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map() for Buffer \""
				<< osg::Object::getName() <<"\": calling thread differs from the context's thread."
				<< std::endl;

			return NULL;
		}

		BufferStream* stream = static_cast<BufferStream*>( osgCompute::Buffer::lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map() for buffer \""
				<< osg::Object::getName() <<"\": could not receive BufferStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return NULL;
		}

		void* ptr = NULL;
		if( mapping != osgCompute::UNMAPPED )
			ptr = mapStream( *stream, mapping );
		else
			unmapStream( *stream );

		return ptr;
	}

	//------------------------------------------------------------------------------
	void Buffer::unmap( const osgCompute::Context& context ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map() for buffer \""
				<< osg::Object::getName() <<"\": buffer is dirty."
				<< std::endl;

			return;
		}

		if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map() for Buffer \""
				<< osg::Object::getName() <<"\": calling thread differs from the context's thread."
				<< std::endl;

			return;
		}

		BufferStream* stream = static_cast<BufferStream*>( osgCompute::Buffer::lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Buffer::map() for buffer \""
				<< osg::Object::getName() <<"\": could not receive BufferStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::Buffer::setMemory( const osgCompute::Context& context, int value, unsigned int mapping, unsigned int offset, unsigned int count ) const
	{
		unsigned char* data = static_cast<unsigned char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setMemory() for buffer \""
					<< osg::Object::getName() <<"\": error during memset() for host within context \""
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
					<< "osgCuda::Buffer::setMemory() for buffer \""
					<< osg::Object::getName() <<"\": error during cudaMemset() for device data within context \""
					<< context.getId() << "\"."
					<< std::endl;


				unmap( context );
				return false;
			}
		}

		return true;
	}

	//------------------------------------------------------------------------------
	void* Buffer::mapStream( BufferStream& stream, unsigned int mapping ) const
	{
		void* ptr = NULL;

		bool needsSetup = false;
		if( (_image.valid() && _image->getModifiedCount() != stream._modifyCount ) ||
			(_array.valid() && _array->getModifiedCount() != stream._modifyCount ) )
			needsSetup = true;

		///////////////////
		// PROOF MAPPING //
		///////////////////
		if( ((stream._mapping & osgCompute::MAP_DEVICE && mapping & osgCompute::MAP_DEVICE) ||
			(stream._mapping & osgCompute::MAP_HOST && mapping & osgCompute::MAP_HOST))  &&
			!needsSetup )
		{
			if( (stream._mapping & osgCompute::MAP_DEVICE) )
				ptr = stream._devPtr;
			else
				ptr = stream._hostPtr;

			if( osgCompute::Buffer::getSubloadResourceCallback() && NULL != ptr )
			{
				const osgCompute::BufferSubloadCallback* callback =
					dynamic_cast<const osgCompute::BufferSubloadCallback*>(osgCompute::Buffer::getSubloadResourceCallback());
				if( callback )
				{
					// subload data before returning the pointer
					callback->subload( ptr, mapping, *this, *stream._context );
				}
			}

			stream._mapping = mapping;
			return ptr;
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
			if( needsSetup )
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
				<< "osgCuda::Buffer::mapStream() for Buffer \""<< osg::Object::getName()<<"\": Wrong mapping. Use one of the following: "
				<< "HOST_SOURCE, HOST_TARGET, HOST, DEVICE_SOURCE, DEVICE_TARGET, DEVICE."
				<< std::endl;

			return NULL;
		}

		//////////////////
		// LOAD/SUBLOAD //
		//////////////////
		if( osgCompute::Buffer::getSubloadResourceCallback() && NULL != ptr )
		{
			const osgCompute::BufferSubloadCallback* callback =
				dynamic_cast<const osgCompute::BufferSubloadCallback*>(osgCompute::Buffer::getSubloadResourceCallback());
			if( callback )
			{
				// load or subload data before returning the host pointer
				if( firstLoad )
					callback->load( ptr, mapping, *this, *stream._context );
				else
					callback->subload( ptr, mapping, *this, *stream._context );
			}
		}

		return ptr;
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
					<< "osgCuda::Buffer::setupStream() for buffer \""<< osg::Object::getName()
					<< "\": cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._devPtr,  data, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""<< osg::Object::getName()
					<< "\": error during cudaMemcpy() within context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
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
					<< "osgCuda::Buffer::setupStream() for buffer \""<< osg::Object::getName()
					<< "\": Cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._hostPtr,  data, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""<< osg::Object::getName()
					<< "\": error during cudaMemcpy() within context \""
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
	bool Buffer::allocStream( unsigned int mapping, BufferStream& stream ) const
	{
		if( mapping & osgCompute::MAP_HOST )
		{
			if( stream._hostPtr != NULL )
				return true;

			if( (stream._allocHint & osgCompute::ALLOC_DYNAMIC) == osgCompute::ALLOC_DYNAMIC )
			{
				stream._hostPtr = static_cast<Context*>(stream._context.get())->mallocDeviceHostMemory( osgCompute::Buffer::getByteSize() );
				if( NULL == stream._hostPtr )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream() for Buffer \""
						<< osg::Object::getName()<<"\": error during mallocDeviceHost() within Context \""<<stream._context->getId()
						<< "\"."
						<< std::endl;

					return false;
				}
			}
			else
			{
				stream._hostPtr = static_cast<Context*>(stream._context.get())->mallocHostMemory( osgCompute::Buffer::getByteSize() );
				if( NULL == stream._hostPtr )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream() for Buffer \""
						<< osg::Object::getName()<<"\": error during mallocHost() within Context \""<<stream._context->getId()
						<< "\"."
						<< std::endl;

					return false;
				}
			}

			stream._hostPtrAllocated = true;
			if( stream._devPtr != NULL )
				stream._syncHost = true;
			return true;
		}
		else if( mapping & osgCompute::MAP_DEVICE )
		{
			if( stream._devPtr != NULL )
				return true;

			if( osgCompute::Buffer::getNumDimensions() == 3 )
			{
				stream._devPtr = static_cast<Context*>(stream._context.get())->mallocDevice3DMemory(
										osgCompute::Buffer::getDimension(0) * getElementSize(),
										osgCompute::Buffer::getDimension(1),
										osgCompute::Buffer::getDimension(2) );

				if( NULL == stream._devPtr )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream() for Buffer \""<< osg::Object::getName()<<"\": error during mallocDevice3D() within Context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}
			}
			else if( osgCompute::Buffer::getNumDimensions() == 2 )
			{
				stream._devPtr = static_cast<Context*>(stream._context.get())->mallocDevice2DMemory(
												osgCompute::Buffer::getDimension(0) * getElementSize(),
												osgCompute::Buffer::getDimension(1));

				if( NULL == stream._devPtr )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream() for Buffer \""<< osg::Object::getName()<<"\": error during mallocDevice2D() within Context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}
			}
			else
			{
				stream._devPtr = static_cast<Context*>(stream._context.get())->mallocDeviceMemory(osgCompute::Buffer::getByteSize());
				if( NULL == stream._devPtr )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Buffer::allocStream() for Buffer \""<< osg::Object::getName()<<"\": error during mallocDevice() within Context \""
						<< stream._context->getId() << "\"."
						<< std::endl;

					return false;
				}
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
			res = cudaMemcpy( stream._devPtr, stream._hostPtr, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::syncStream() for Buffer \""<< osg::Object::getName()
					<< "\": error during cudaMemcpy() to device within Context \""
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
			res = cudaMemcpy( stream._hostPtr, stream._devPtr, osgCompute::Buffer::getByteSize(), cudaMemcpyDeviceToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::syncStream() for Buffer \""
					<< osg::Object::getName()<<"\": error during cudaMemcpy() to host within Context \""
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
	void Buffer::setImage( osg::Image* image )
	{
		if( !osgCompute::Resource::isClear() && image != NULL )
		{
			if( image->getNumMipmapLevels() > 1 )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""
					<< osg::Object::getName() <<"\": image \""
					<< image->getName() << "\" uses MipMaps which are currently"
					<< "not supported."
					<< std::endl;

				return;
			}

			if( image->getTotalSizeInBytes() != osgCompute::Buffer::getByteSize() )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""
					<< osg::Object::getName() <<"\": size of image \""
					<< image->getName() << "\" is wrong."
					<< std::endl;

				return;
			}
		}

		_image = image;
		_array = NULL;
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
			if( array->getTotalDataSize() != osgCompute::Buffer::getByteSize() )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setArray() for buffer \""
					<< osg::Object::getName() <<"\": size of array \""
					<< array->getName() << "\" is wrong."
					<< std::endl;

				return;
			}
		}

		_array = array;
		_image = NULL;
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
}
