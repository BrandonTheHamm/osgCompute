#include <memory.h>
#include <osg/GL>
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
		{
			cudaError_t res = cudaGLUnregisterBufferObject( _bo );
			if( res != cudaSuccess )
				osg::notify(osg::FATAL)
					<<"TextureStream::~TextureStream(): error during cudaGLUnregisterBufferObject()."
					<< cudaGetErrorString(res) << std::endl;
		}
		
		if( _bo != UINT_MAX )
		{
			osg::State* state = _context->getGraphicsContext()->getState();
			if( state )
			{
				// ... delete buffer object
				osg::BufferObject::Extensions* ext = osg::BufferObject::getExtensions( state->getContextID(),true);
				ext->glBindBuffer( GL_ARRAY_BUFFER_ARB, _bo );
				ext->glDeleteBuffers( 1, &_bo );
				GLenum errorStatus = glGetError();
				if( errorStatus != GL_NO_ERROR )
					osg::notify(osg::FATAL)
					<<"TextureStream::~TextureStream(): error during glDeleteBuffers()."<<std::endl;
				ext->glBindBuffer( GL_ARRAY_BUFFER_ARB, 0 );
			}
		}

		if( _hostPtrAllocated && NULL != _hostPtr)
			free( _hostPtr );
	}



    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	TextureBuffer::TextureBuffer()
		: osgCompute::InteropBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	TextureBuffer::~TextureBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void TextureBuffer::clear()
	{
		osgCompute::Buffer::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool TextureBuffer::init()
	{
		if( !isClear() )
			return true;

		if( !asTexture() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureBuffer::init(): object must be of type osg::Texture."
				<< std::endl;

			clear();
			return false;
		}

		if( !initElementSize() )
		{
			clear();
			return false;
		}

		if( !initDimension() )
		{
			clear();
			return false;
		}

		return osgCompute::Buffer::init();
	}

	//------------------------------------------------------------------------------
	bool TextureBuffer::initDimension()
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

		if( dim[0] == 0 )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture2DBuffer::initDimension(): no dimensions defined for texture! set the texture dimensions first."
				<< std::endl;

			return false;
		}

		unsigned int d = 0;
		while( dim[d] > 0 && d < 3 )
		{
			setDimension( d, dim[d] );
			++d;
		}

		return true;
	}

	//------------------------------------------------------------------------------
	bool TextureBuffer::initElementSize()
	{
		unsigned int elementSize = 0;
		unsigned int elementBitSize;
		if( asTexture()->getImage(0) )
		{
			elementBitSize = 
				osg::Image::computePixelSizeInBits( 
				asTexture()->getImage(0)->getPixelFormat(), 
				asTexture()->getImage(0)->getDataType() );

		}
		else
		{
			elementBitSize = 
				osg::Image::computePixelSizeInBits(
				asTexture()->getInternalFormat(), 
				asTexture()->getSourceType() );
		}

		elementSize = ((elementBitSize % 8) == 0)? elementBitSize/8 : elementBitSize/8 +1;
		if( elementSize == 0 )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture2DBuffer::initElementSize(): cannot determine element size."
				<< std::endl;

			return false;
		}

		setElementSize( elementSize ); 
		return true;
	}

	//------------------------------------------------------------------------------
	void* TextureBuffer::map( const osgCompute::Context& context, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "TextureBuffer::map(): buffer is dirty."
				<< std::endl;

			return NULL;
		}

		if( !context.isConnectedWithGraphicsContext() )
		{
			osg::notify(osg::FATAL)
				<< "TextureBuffer::map(): context must be connected to graphics context."
				<< std::endl;

			return NULL;
		}

		TextureStream* stream = static_cast<TextureStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "TextureBuffer::map(): could not receive TextureStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return NULL;
		}


		void* ptr = NULL;
		if( mapping != osgCompute::UNMAPPED )
			ptr = mapStream( *stream, mapping, offset );
		else
			unmapStream( *stream );

		return ptr;
	}

	//------------------------------------------------------------------------------
	void TextureBuffer::unmap( const osgCompute::Context& context, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureBuffer::map(): buffer is dirty."
				<< std::endl;

			return;
		}

		TextureStream* stream = static_cast<TextureStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureBuffer::map(): could not receive TextureStream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::TextureBuffer::setMemory( const osgCompute::Context& context, int value, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int count, unsigned int ) const
	{
		unsigned char* data = static_cast<unsigned char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::setMemory(): error during memset() for host within context \""
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
					<< "osgCuda::TextureBuffer::setMemory(): error during cudaMemset() for device data within context \""
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

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void TextureBuffer::clearLocal()
	{
	}

	//------------------------------------------------------------------------------
	void TextureBuffer::clear( const osgCompute::Context& context ) const
	{
		if( getMapping( context ) != osgCompute::UNMAPPED )
			unmap( context );

		osgCompute::Buffer::clear( context );
	}

	//------------------------------------------------------------------------------
	void* TextureBuffer::mapStream( TextureStream& stream, unsigned int mapping, unsigned int offset ) const
	{
		void* ptr = NULL;

		///////////////////
		// PROOF MAPPING //
		///////////////////
		if( !getIsRenderTarget() && 
			((stream._mapping & osgCompute::MAP_DEVICE && mapping & osgCompute::MAP_DEVICE ) ||
			(stream._mapping & osgCompute::MAP_HOST && mapping & osgCompute::MAP_HOST )) )
		{
			if( (stream._mapping & osgCompute::MAP_DEVICE) )
				ptr = stream._devPtr;
			else
				ptr = stream._hostPtr;


			if( getSubloadCallback() && NULL != ptr )
			{
				const osgCompute::SubloadCallback* callback = getSubloadCallback();
				if( callback )
				{
					// subload data before returning the pointer
					callback->subload( ptr, mapping, offset, *this, *stream._context );
				}
			}

			stream._mapping = mapping;
			return &static_cast<char*>(ptr)[offset];
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
		if( getIsRenderTarget() && !(mapping & osgCompute::MAP_DEVICE_TARGET) )
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
					<< "osgCuda::TextureBuffer::mapStream(): error during cudaGLMapBufferObject()."
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
			if( asTexture()->getImage(0) && asTexture()->getImage(0)->getModifiedCount() != stream._modifyCount )
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
			if( asTexture()->getImage(0) && asTexture()->getImage(0)->getModifiedCount() != stream._modifyCount )
				if( !setupStream( mapping, stream ) )
					return NULL;

			// sync stream with device is already done at this point.
			// See unmapStream().
			ptr = stream._devPtr;
		}
		else
		{
			osg::notify(osg::WARN)
				<< "osgCuda::TextureBuffer::mapStream(): wrong mapping type specified. Use one of the following types: "
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

		stream._mapping = mapping;
		return &static_cast<char*>(ptr)[offset];
	}

	//------------------------------------------------------------------------------
	void TextureBuffer::unmapStream( TextureStream& stream ) const
	{
		if( stream._mapping & osgCompute::MAP_HOST_TARGET )
		{
			// synchronize CPU memory with PBO. This cannot wait
			// since the texture might be applied to a state.
			syncStream( osgCompute::MAP_DEVICE, stream );
		}

		///////////
		// UNMAP //
		///////////
		if( stream._devPtr != NULL )
		{
			cudaError res = cudaGLUnmapBufferObject( stream._bo );
			if( cudaSuccess != res )
			{
				osg::notify(osg::WARN)
					<< "osgCuda::TextureBuffer::unmapStream(): error during cudaGLUnmapBufferObject()."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;
				return;
			}
			stream._devPtr = NULL;
		}

		////////////////////
		// UPDATE TEXTURE //
		////////////////////
		if( stream._mapping & osgCompute::MAP_DEVICE_TARGET )
		{
			// sync texture object as required
			syncTexture( stream );
			stream._syncHost = true;
		}

		stream._mapping = osgCompute::UNMAPPED;
	}

	//------------------------------------------------------------------------------
	bool TextureBuffer::setupStream( unsigned int mapping, TextureStream& stream ) const
	{
		if( !asTexture()->getImage(0) )
			return true;

		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			const void* data = asTexture()->getImage(0)->data();

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::setupStream(): Cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._devPtr,  data, getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::setupStream(): error during cudaMemcpy()."
					<< " " << cudaGetErrorString( res ) << "."
					<< std::endl;

				return false;
			}

			// host must be synchronized
			stream._syncHost = true;
			stream._modifyCount = asTexture()->getImage(0)->getModifiedCount();

			return true;
		}
		else if( mapping & osgCompute::MAP_HOST )
		{
			const void* data = asTexture()->getImage(0)->data();

			if( data == NULL )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::setupStream(): Cannot receive valid data pointer."
					<< std::endl;

				return false;
			}

			res = cudaMemcpy( stream._hostPtr,  data, getByteSize(), cudaMemcpyHostToHost );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::setupStream(): error during cudaMemcpy()."
					<< " " << cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			// device must be synchronized
			stream._syncDevice = true;
			stream._modifyCount = asTexture()->getImage(0)->getModifiedCount();

			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool TextureBuffer::allocStream( unsigned int mapping, TextureStream& stream ) const
	{
		if( mapping & osgCompute::MAP_HOST )
		{
			if( stream._hostPtr != NULL )
				return true;


			stream._hostPtr = malloc( getByteSize() );
			if( NULL == stream._hostPtr )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::allocStream(): something goes wrong within mallocHost()."
					<< std::endl;

				return false;
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
	bool TextureBuffer::syncStream( unsigned int mapping, TextureStream& stream ) const
	{
		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			res = cudaMemcpy( stream._devPtr, stream._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureBuffer::syncStream(): error during cudaMemcpy() to device."
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
					<< "osgCuda::TextureBuffer::syncStream(): something goes wrong within cudaMemcpy() to host."
					<< " " << cudaGetErrorString( res ) << "."
					<< std::endl;

				return false;
			}

			stream._syncHost = false;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	osgCompute::BufferStream* TextureBuffer::newStream( const osgCompute::Context& context ) const
	{
		return new TextureStream;
	}
}
