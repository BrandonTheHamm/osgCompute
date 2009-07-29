#include <memory.h>
#include <osg/GL>
#include <osg/Texture>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/Texture>

namespace osgCuda
{
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	TextureStream::TextureStream()
		: osgCompute::BufferStream(),
		_devPtr(NULL),
		_hostPtr(NULL),
		_syncDevice(false),
		_syncHost(false),
		_hostPtrAllocated(false),
		_bo( UINT_MAX ),
		_boRegistered(false),
		_modifyCount(UINT_MAX)
	{
	}

	//------------------------------------------------------------------------------
	TextureStream::~TextureStream()
	{
		if( _boRegistered && _bo != UINT_MAX )
			static_cast<Context*>( osgCompute::BufferStream::_context.get() )->freeBufferObject( _bo );
		if( _hostPtrAllocated && NULL != _hostPtr)
			static_cast<Context*>(osgCompute::BufferStream::_context.get())->freeMemory( _hostPtr );
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture::Texture()
		: osgCompute::Buffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	Texture::~Texture()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void Texture::clear()
	{
		osgCompute::Buffer::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool Texture::init()
	{
		if( !osgCompute::Resource::isClear() )
			return true;

		if( !asTexture() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture::init() for buffer \""
				<< asObject()->getName() <<"\": object must be of type osg::Texture."
				<< std::endl;

			clear();
			return false;
		}

		if( !getElementSize() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture::init() for buffer \""
				<< asObject()->getName() <<"\": no element size specified."
				<< std::endl;

			clear();
			return false;
		}

		// some flags for textures are not available right now
		// like resize to a power of two and mipmaps
		asTexture()->setResizeNonPowerOfTwoHint( false );
		asTexture()->setUseHardwareMipMapGeneration( false );

		if( !initDimension() )
		{
			clear();
			return false;
		}

		//////////////////////
		// COMPUTE BYTE SIZE //
		///////////////////////
		unsigned int numElements = 1;
		for( unsigned int d=0; d<osgCompute::Buffer::_dimensions.size(); ++d )
			numElements *= osgCompute::Buffer::_dimensions[d];

		/////////////////////
		// CHECK BYTE SIZE //
		/////////////////////
		GLenum texType = GL_NONE;
		GLenum texFormat = GL_NONE;
		if( asTexture()->getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT )
		{
			osg::Image* image = asTexture()->getImage(0);
			if(!image)
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture::init() for texture \""
					<< asObject()->getName()<< "\": wrong internal format."
					<< std::endl;

				return false;
			}

			texType = image->getDataType();
		}
		else
		{
			texType = (asTexture()->getSourceType() != GL_NONE)? asTexture()->getSourceType() : GL_UNSIGNED_BYTE;
		}

		unsigned int bitSize = osg::Image::computePixelSizeInBits( asTexture()->getInternalFormat(), texType );
		unsigned int byteSize = ((bitSize % 8) == 0)? (bitSize/8) : (bitSize/8+1);

		if( byteSize != getElementSize() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture::init() for texture \""
				<< asObject()->getName()<< "\": wrong element size specified."
				<< std::endl;

			return false;
		}

		return osgCompute::Buffer::init();
	}

	//------------------------------------------------------------------------------
	bool Texture::initDimension()
	{
		unsigned int dim[3];
		if( asTexture()->getImage(0) )
		{
			dim[0] = asTexture()->getImage(0)->s();
			dim[1] = asTexture()->getImage(0)->t();
			dim[2] = asTexture()->getImage(0)->r();
		}
		else
		{
			dim[0] = asTexture()->getTextureWidth();
			dim[1] = asTexture()->getTextureHeight();
			dim[2] = asTexture()->getTextureDepth();
		}

		if( osgCompute::Buffer::getNumDimensions() == 0 )
		{
			if( dim[0] == 0 )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture::initDimension() for texture \""
					<< asObject()->getName()<< "\": no dimensions defined for texture! Call setDimension() first."
					<< std::endl;

				return false;
			}

			unsigned int d = 0;
			while( dim[d] > 0 )
			{
				osgCompute::Buffer::setDimension( d, dim[d] );
				++d;
			}
		}
		else
		{
			for( unsigned int d=0; d<osgCompute::Buffer::getNumDimensions(); ++d )
			{
				if( dim[d] > 0 && dim[d] != osgCompute::Buffer::getDimension(d) )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Texture::initDimension() for texture \""
						<< asObject()->getName()<< "\": different dimensions for cuda context and rendering context specified!"
						<< std::endl;

					return false;
				}
			}
		}

		return true;
	}

	//------------------------------------------------------------------------------
	void* Texture::map( const osgCompute::Context& context, unsigned int mapping ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "Texture::map() for texture \""
				<< asObject()->getName() <<"\": buffer is dirty."
				<< std::endl;

			return NULL;
		}

		if( getIsRenderTarget() &&
			( ((mapping & osgCompute::MAP_DEVICE)  == osgCompute::MAP_DEVICE_TARGET )  ||
			((mapping & osgCompute::MAP_HOST)  == osgCompute::MAP_HOST_TARGET) ) )
		{
			osg::notify(osg::WARN)
				<< "Texture::map() for texture \""<< asObject()->getName()
				<< "\": texture is target of a compute context and target of a render context. This is not allowed."
				<< std::endl;

			return NULL;
		}

		if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		{
			osg::notify(osg::FATAL)
				<< "Texture::map() for texture \""
				<< asObject()->getName() <<"\": calling thread differs from the context's thread."
				<< std::endl;

			return NULL;
		}

		TextureStream* stream = static_cast<TextureStream*>( osgCompute::Buffer::lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "Texture::map() for texture \""
				<< asObject()->getName() <<"\": could not receive TextureStream for context \""
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
	void Texture::unmap( const osgCompute::Context& context ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture::map() for texture \""
				<< asObject()->getName() <<"\": buffer is dirty."
				<< std::endl;

			return;
		}

		if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture::map() for texture \""
				<< asObject()->getName() <<"\": calling thread differs from the context's thread."
				<< std::endl;

			return;
		}

		TextureStream* stream = static_cast<TextureStream*>( osgCompute::Buffer::lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture::map() for texture \""
				<< asObject()->getName() <<"\": could not receive TextureStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::Texture::setMemory( const osgCompute::Context& context, int value, unsigned int mapping, unsigned int offset, unsigned int count ) const
	{
		unsigned char* data = static_cast<unsigned char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture::setMemory() for texture \""
					<< asObject()->getName() <<"\": error during memset() for host within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}

			return true;
		}
		else if( mapping & osgCompute::MAP_DEVICE_TARGET )
		{
			cudaError res = cudaMemset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count );
			if( res != cudaSuccess )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture::setMemory() for texture \""
					<< asObject()->getName() <<"\": error during cudaMemset() for device data within context \""
					<< context.getId() << "\"."
					<< std::endl;

				unmap( context );
				return false;
			}

			return true;
		}

		unmap( context );
		return false;
	}

	//------------------------------------------------------------------------------
	osg::Image* Texture::getImagePtr()
	{
		return NULL;
	}


	//------------------------------------------------------------------------------
	const osg::Image* Texture::getImagePtr() const
	{
		return NULL;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture::clearLocal()
	{
		_isRenderTarget = false;
	}

	//------------------------------------------------------------------------------
	void Texture::clear( const osgCompute::Context& context ) const
	{
		if( osgCompute::Buffer::getMapping( context ) != osgCompute::UNMAPPED )
			unmap( context );

		osgCompute::Buffer::clear( context );
	}

	//------------------------------------------------------------------------------
	void* Texture::mapStream( TextureStream& stream, unsigned int mapping ) const
	{
		void* ptr = NULL;

		///////////////////
		// PROOF MAPPING //
		///////////////////
		if( (stream._mapping & osgCompute::MAP_DEVICE && mapping & osgCompute::MAP_DEVICE ) ||
			(stream._mapping & osgCompute::MAP_HOST && mapping & osgCompute::MAP_HOST ) )
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


		bool firstLoad = false;

		////////////////////////////
		// ALLOCATE DEVICE-MEMORY //
		////////////////////////////
		// create dynamic texture device memory
		// for each type of mapping
		if( UINT_MAX == stream._bo )
		{
			if( !allocStream( osgCompute::MAP_DEVICE, stream ) )
				return NULL;

			firstLoad = true;
		}

		////////////////
		// UPDATE PBO //
		////////////////
		// if necessary sync dynamic texture with PBO
		if( getIsRenderTarget() )
		{
			// synchronize PBO with texture
			syncPBO( stream );
			stream._syncHost = true;
		}

		/////////////
		// MAP PBO //
		/////////////
		if( NULL == stream._devPtr )
		{
			cudaError res = cudaGLMapBufferObject( &stream._devPtr, stream._bo );
			if( cudaSuccess != res )
			{
				osg::notify(osg::WARN)
					<< "osgCuda::Texture::mapStream() for texture \""<< asObject()->getName()
					<< "\": error during cudaGLMapBufferObject() for context \""
					<< stream._context->getId()<<"\"."
					<< " " << cudaGetErrorString( res ) << "."
					<< std::endl;

				return NULL;
			}
		}

		//////////////
		// MAP DATA //
		//////////////

		if( mapping & osgCompute::MAP_HOST )
		{
			if( NULL == stream._hostPtr )
			{
				////////////////////////////
				// ALLOCATE DEVICE-MEMORY //
				////////////////////////////
				if( !allocStream( mapping, stream ) )
					return NULL;
			}

			//////////////////
			// SETUP STREAM //
			//////////////////
			if( getImagePtr() && getImagePtr()->getModifiedCount() != stream._modifyCount )
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
		else if( (mapping & osgCompute::MAP_DEVICE) )
		{
			//////////////////
			// SETUP STREAM //
			//////////////////
			if( getImagePtr() && getImagePtr()->getModifiedCount() != stream._modifyCount )
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
				<< "osgCuda::Texture::mapStream() for texture \""<< asObject()->getName()<<"\": wrong mapping type. Use one of the following types: "
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

		stream._mapping = mapping;
		return ptr;
	}

	//------------------------------------------------------------------------------
	void Texture::unmapStream( TextureStream& stream ) const
	{
		///////////
		// UNMAP //
		///////////
		if( stream._devPtr != NULL )
		{
			cudaError res = cudaGLUnmapBufferObject( stream._bo );
			if( cudaSuccess != res )
			{
				osg::notify(osg::WARN)
					<< "osgCuda::Texture::unmapStream() for texture \""<< asObject()->getName()
					<<"\": error during cudaGLUnmapBufferObject() for context \""
					<< stream._context->getId()<<"\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;
				return;
			}
			stream._devPtr = NULL;
		}

		////////////////////
		// UPDATE TEXTURE //
		////////////////////
		if( stream._mapping == osgCompute::MAP_DEVICE_TARGET )
		{
			// sync texture object as required
			syncTexture( stream );
			stream._syncHost = true;
		}
		else if( (stream._mapping & osgCompute::MAP_HOST_TARGET) )
		{
			stream._syncDevice = true;
		}

		stream._mapping = osgCompute::UNMAPPED;
	}

	//------------------------------------------------------------------------------
	bool Texture::setupStream( unsigned int mapping, TextureStream& stream ) const
	{
		if( !getImagePtr() )
			return true;

		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			const void* data = getImagePtr()->data();

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""<< asObject()->getName()
					<< "\": Cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._devPtr,  data, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""<< asObject()->getName()
					<< "\": error during cudaMemcpy() within context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) << "."
					<< std::endl;

				return false;
			}

			// host must be synchronized
			stream._syncHost = true;
			stream._modifyCount = getImagePtr()->getModifiedCount();

			return true;
		}
		else if( mapping & osgCompute::MAP_HOST )
		{
			const void* data = getImagePtr()->data();

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""<< asObject()->getName()
					<< "\": Cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._hostPtr,  data, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Buffer::setupStream() for buffer \""<< asObject()->getName()
					<< "\": error during cudaMemcpy() within context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			// device must be synchronized
			stream._syncDevice = true;
			stream._modifyCount = getImagePtr()->getModifiedCount();

			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Texture::allocStream( unsigned int mapping, TextureStream& stream ) const
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
						<< "osgCuda::Texture::allocStream() for texture \""
						<< asObject()->getName()<<"\": something goes wrong within mallocDeviceHost() within context \""<<stream._context->getId()
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
						<< "osgCuda::Texture::allocStream() for texture \""
						<< asObject()->getName()<<"\": something goes wrong within mallocHost() within context \""<<stream._context->getId()
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
			if( stream._bo != UINT_MAX )
				return true;

			allocPBO( stream );

			//////////////
			// INIT PBO //
			//////////////
			syncPBO( stream );

			if( stream._hostPtr != NULL )
				stream._syncDevice = true;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Texture::syncStream( unsigned int mapping, TextureStream& stream ) const
	{
		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			res = cudaMemcpy( stream._devPtr, stream._hostPtr, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture::syncStream() for texture \""<< asObject()->getName()
					<< "\": error during cudaMemcpy() to device within context \""
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
					<< "osgCuda::Texture::syncStream() for texture \""
					<< asObject()->getName()<<"\": something goes wrong within cudaMemcpy() to host within context \""
					<< stream._context->getId() << "\"."
					<< " " << cudaGetErrorString( res ) << "."
					<< std::endl;

				return false;
			}

			stream._syncHost = false;
			return true;
		}

		return false;
	}
}

