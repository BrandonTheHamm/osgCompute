#include <malloc.h>
#include <memory.h>
#include <osg/GL>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/Texture>

namespace osgCuda
{
	/**
	*/
	class Texture3DBuffer : public osgCuda::TextureBuffer
	{
	public:
		Texture3DBuffer();

		META_Object(osgCuda,Texture3DBuffer)

		virtual bool init();

		virtual osgCompute::InteropObject* getObject() { return _texref.get(); }
		virtual osg::Texture* asTexture() { return _texref.get(); }
		virtual const osg::Texture* asTexture() const { return _texref.get(); }

		virtual bool getIsRenderTarget() const;

		virtual void clear( const osgCompute::Context& context ) const;
		virtual void clear();
	protected:
		friend class Texture3D;
		virtual ~Texture3DBuffer();
		void clearLocal();

		virtual void setIsRenderTarget( bool isRenderTarget );
		virtual void syncModifiedCounter( const osgCompute::Context& context ) const;
		virtual bool allocPBO( TextureStream& stream ) const;
		virtual void syncPBO( TextureStream& stream ) const;
		virtual void syncTexture( TextureStream& stream ) const;

		osg::ref_ptr<osgCuda::Texture3D> _texref;
		bool							 _isRenderTarget;
	private:
		// copy constructor and operator should not be called
		Texture3DBuffer( const Texture3DBuffer& , const osg::CopyOp& ) {}
		Texture3DBuffer& operator=(const Texture3DBuffer&) { return (*this); }
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture3DBuffer::Texture3DBuffer()
		: osgCuda::TextureBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	Texture3DBuffer::~Texture3DBuffer()
	{
		clearLocal();

		// proxy is now deleted
		_texref->_proxy = NULL;
		// reattach handles
		_texref->_handles = getHandles();
		// decrease reference count of texture reference
		// in order to release texture object if
		// object was only referenced within the computation part
		_texref = NULL;
	}

	//------------------------------------------------------------------------------
	void Texture3DBuffer::clear()
	{
		osgCuda::TextureBuffer::clear();
		clearLocal();
	}

	void Texture3DBuffer::clear( const osgCompute::Context& context ) const
	{
		TextureBuffer::clear( context );
	}

	//------------------------------------------------------------------------------
	bool Texture3DBuffer::init()
	{
		if( !isClear() )
			return true;

		if( !_texref.valid() )
			return false;

		_isRenderTarget = _texref->getIsRenderTarget();

		return osgCuda::TextureBuffer::init();
	}

	//------------------------------------------------------------------------------
	bool Texture3DBuffer::getIsRenderTarget() const
	{
		return _isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void Texture3DBuffer::setIsRenderTarget( bool isRenderTarget )
	{
		_isRenderTarget = isRenderTarget;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture3DBuffer::clearLocal()
	{
		_isRenderTarget = false;
	}

	//------------------------------------------------------------------------------
	void Texture3DBuffer::syncModifiedCounter( const osgCompute::Context& context ) const
	{
		if( !_texref->getImage() || !context.isConnectedWithGraphicsContext() )
			return;

		const osg::State* state = context.getGraphicsContext()->getState();
		if( state == NULL )
			return;

		_texref->getModifiedCount(state->getContextID()) = _texref->getImage()->getModifiedCount();
	}

	//------------------------------------------------------------------------------
	void Texture3DBuffer::syncTexture( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return;

		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( state->getContextID(),true );
		osg::Texture3D::Extensions* tex3DExt = osg::Texture3D::getExtensions( state->getContextID(),true );
		if( !bufferExt || !tex3DExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncTexture(): cannot find required extension."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncTexture(): texture object not allocated."
				<< std::endl;

			return;
		}

		GLenum texType = GL_NONE;
		if( asTexture()->getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT )
		{
			if( !_texref->getImage() )
				return;

			texType = _texref->getImage()->getDataType();
		}
		else
		{
			texType = asTexture()->getSourceType();
		}

		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB,  stream._bo );
		glBindTexture( GL_TEXTURE_3D, tex->_id );

		// UNPACK the PBO data
		tex3DExt->glTexSubImage3D(
			GL_TEXTURE_3D, 0, 0, 0, 0,
			getDimension(0),
			getDimension(1),
			getDimension(2),
			tex->_internalFormat,
			texType,
			NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncTexture(): error occured during glTex(Sub)Image3D(). Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}


		glBindTexture( GL_TEXTURE_3D, 0 );
		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	}

	//------------------------------------------------------------------------------
	void Texture3DBuffer::syncPBO( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;


		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return;

		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( state->getContextID(),true );
		osg::Texture3D::Extensions* tex3DExt = osg::Texture3D::getExtensions( state->getContextID(),true );
		if( !bufferExt || !tex3DExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncPBO(): cannot find required extension."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncPBO() : texture object not allocated."
				<< std::endl;

			return;
		}

		GLenum texType = GL_NONE;
		if( asTexture()->getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT &&
			_texref->getImage() )
		{
			texType = _texref->getImage()->getDataType();
		}
		else
		{
			texType = asTexture()->getSourceType();
		}

		if( texType == GL_NONE )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncPBO(): texture type unknown."
				<< std::endl;

			return;
		}

		////////////////////
		// UNREGISTER PBO //
		////////////////////
		cudaError res = cudaGLUnregisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncPBO(): error during cudaGLUnregisterBufferObject(). " << cudaGetErrorString( res ) <<"."
				<< std::endl;

			return;
		}

		///////////////
		// COPY DATA //
		///////////////
		glBindTexture( GL_TEXTURE_3D, tex->_id );
		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB,  stream._bo );

		// PACK the data for the PBO
		glGetTexImage( GL_TEXTURE_3D, 0, tex->_internalFormat, texType, NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncPBO(): error during glGetTexImage(). Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}

		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, 0 );
		glBindTexture( GL_TEXTURE_3D, 0 );

		////////////////////////
		// REGISTER PBO AGAIN //
		////////////////////////
		res = cudaGLRegisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::syncPBO(): error during cudaGLRegisterBufferObject()." << cudaGetErrorString( res ) << "."
				<< std::endl;
		}

		stream._syncHost = true;
	}

	//------------------------------------------------------------------------------
	bool Texture3DBuffer::allocPBO( TextureStream& stream ) const
	{
		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return false;

		/////////////////////
		// COMPILE TEXTURE //
		/////////////////////
		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( state->getContextID(),true );
		osg::Texture3D::Extensions* tex3DExt = osg::Texture3D::getExtensions( state->getContextID(),true );
		if( !bufferExt || !tex3DExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): cannot find required extension."
				<< std::endl;

			return false;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			_texref->apply( *state );
			glBindTexture( GL_TEXTURE_3D, 0 );
			GLenum errorStatus = glGetError();
			if( errorStatus != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture3DBuffer::allocPBO(): apply() failed on texture resource. Maybe context is not active."
					<< std::endl;

				return false;
			}

			// second chance
			tex = asTexture()->getTextureObject( state->getContextID() );
		}

		if( !tex )
		{

			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): cannot allocate texture object. Maybe context is not active."
				<< std::endl;

			return false;
		}

		GLenum texType = GL_NONE;
		if( asTexture()->getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT &&
			_texref->getImage() )
		{
			texType = _texref->getImage()->getDataType();
		}
		else
		{
			texType = asTexture()->getSourceType();
		}

		if( texType == GL_NONE )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): texture type unknown."
				<< std::endl;

			return false;
		}

		///////////////
		// ALLOC PBO //
		///////////////
		bufferExt->glGenBuffers( 1, &stream._bo );
		GLenum errorNo = glGetError();
		if( 0 == stream._bo  || errorNo )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): cannot generate BufferObject (glGenBuffers())."
				<< std::endl;

			return UINT_MAX;
		}

		////////////////////
		// INITIALIZE PBO //
		////////////////////
		// Allocate temporary memory
		void *tmpData = malloc(getByteSize());
		memset( tmpData, 0x0, getByteSize() );

		// Initialize PixelBufferObject
		bufferExt->glBindBuffer( GL_ARRAY_BUFFER_ARB, stream._bo );
		errorNo = glGetError();
		if (errorNo != GL_NO_ERROR)
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): cannot bind BufferObject (glBindBuffer())."
				<< std::endl;

			return UINT_MAX;
		}

		bufferExt->glBufferData( GL_ARRAY_BUFFER_ARB, getByteSize(), tmpData, GL_DYNAMIC_DRAW );
		errorNo = glGetError();
		if (errorNo != GL_NO_ERROR)
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): cannot initialize BufferObject (glBufferData())."
				<< std::endl;

			return UINT_MAX;
		}
		bufferExt->glBindBuffer( GL_ARRAY_BUFFER_ARB, 0 );

		// Free temporary memory
		if( tmpData )
			free(tmpData);


		//////////////////
		// REGISTER PBO //
		//////////////////
		cudaError res = cudaGLRegisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): error during cudaGLRegisterBufferObject()."
				<< cudaGetErrorString(res) << "."
				<< std::endl;

			return UINT_MAX;
		}

		if( UINT_MAX == stream._bo )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3DBuffer::allocPBO(): cannot generate buffer object (glGenBuffers())."
				<< std::endl;

			return false;
		}
		stream._boRegistered = true;

		///////////////////
		// SETUP TEXTURE //
		///////////////////
		if( tex->isAllocated() )
		{
			//////////////
			// SYNC PBO //
			//////////////
			// Sync PBO with texture-data
			// if it is already allocated
			syncPBO( stream );
		}
		else if( !this->getIsRenderTarget() )
		{
			/////////////////////
			// ALLOCATE MEMORY //
			/////////////////////
			glBindTexture( GL_TEXTURE_3D, tex->_id );

			tex3DExt->glTexImage3D(
				GL_TEXTURE_3D, 0,
				tex->_internalFormat,
				tex->_width, tex->_height, tex->_depth,
				tex->_border,
				tex->_internalFormat, texType, NULL );

			GLenum errorNo = glGetError();
			if( errorNo != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture3D::allocPBO(): error during glTexImage3D(). Returned code is "
					<< std::hex<<errorNo<<"."
					<< std::endl;

				glBindTexture( GL_TEXTURE_3D, 0 );
				bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
				return false;
			}

			glBindTexture( GL_TEXTURE_3D, 0 );

			// set allocated flag for texture object
			tex->setAllocated( true );
		}

		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture3D::Texture3D()
		: osg::Texture3D(),
		  _proxy(NULL)
	{
		// some flags for textures are not available right now
		// like resize to a power of two and mipmaps
		asTexture()->setResizeNonPowerOfTwoHint( false );
		asTexture()->setUseHardwareMipMapGeneration( false );
	}

	//------------------------------------------------------------------------------
	Texture3D::~Texture3D()
	{
		if( _proxy != NULL )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::destructor(): proxy is still valid!!!."
				<< std::endl;
		}
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Texture3D::getBuffer()
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	const osgCompute::InteropBuffer* Texture3D::getBuffer() const
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Texture3D::getOrCreateBuffer()
	{
		// create proxy buffer on demand
		if( _proxy == NULL )
		{
			_proxy = new Texture3DBuffer;
			_proxy->_texref = this;
			_proxy->setHandles( _handles );
			_handles.clear();
			if( !_proxy->init() )
			{
				_proxy->unref();
				_proxy = NULL;
			}
		}

		return _proxy;
	}

	//------------------------------------------------------------------------------
	void Texture3D::addHandle( const std::string& handle )
	{
		if( _proxy != NULL )
		{
			_proxy->addHandle( handle );
		}
		else
		{
			if( !isAddressedByHandle(handle) )
				_handles.insert( handle );
		}
	}

	//------------------------------------------------------------------------------
	void Texture3D::removeHandle( const std::string& handle )
	{
		if( _proxy != NULL )
		{
			_proxy->removeHandle( handle );
		}
		else
		{
			osgCompute::HandleSetItr itr = _handles.find( handle );
			if( itr != _handles.end() )
				_handles.erase( itr );
		}
	}

	//------------------------------------------------------------------------------
	bool Texture3D::isAddressedByHandle( const std::string& handle ) const
	{
		if( _proxy != NULL )
		{
			return _proxy->isAddressedByHandle( handle );
		}
		else
		{
			osgCompute::HandleSetCnstItr itr = _handles.find( handle );
			if( itr == _handles.end() )
				return false;

			return true;
		}
	}

	//------------------------------------------------------------------------------
	void Texture3D::releaseGLObjects( osg::State* state/*=0*/ ) const
	{
		if( state != NULL )
		{
			const osgCompute::Context* curCtx = osgCompute::Context::getContext( state->getContextID() );
			if( curCtx )
			{
				if( NULL != _proxy )
					_proxy->clear( *curCtx );
			}
		}

		osg::Texture3D::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void Texture3D::apply( osg::State& state ) const
	{
		const osgCompute::Context* curCtx = osgCompute::Context::getContext( state.getContextID() );
		if( curCtx )
		{
			if( NULL != _proxy && _proxy->getMapping( *curCtx ) != osgCompute::UNMAPPED )
				_proxy->unmap( *curCtx );
		}

		osg::Texture3D::apply( state );
	}

	//------------------------------------------------------------------------------
	bool Texture3D::getIsRenderTarget() const
	{
		return _isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void Texture3D::setIsRenderTarget( bool isRenderTarget )
	{
		_isRenderTarget = isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void Texture3D::clear()
	{
		clearLocal();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture3D::clearLocal()
	{
		_isRenderTarget = false;
		if( NULL != _proxy )
			_proxy->clear();

		_handles.clear();
	}
}
