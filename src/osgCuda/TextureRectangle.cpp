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
	class TextureRectangleBuffer : public osgCuda::TextureBuffer
	{
	public:
		TextureRectangleBuffer();

		META_Object(osgCuda,TextureRectangleBuffer)

		virtual bool init();

		virtual osgCompute::InteropObject* getObject() { return _texref.get(); }
		virtual osg::Texture* asTexture() { return _texref.get(); }
		virtual const osg::Texture* asTexture() const { return _texref.get(); }

		virtual bool getIsRenderTarget() const;

		virtual void clear( const osgCompute::Context& context ) const;
		virtual void clear();
	protected:
		friend class TextureRectangle;
		virtual ~TextureRectangleBuffer();
		void clearLocal();

		virtual void setIsRenderTarget( bool isRenderTarget );
		virtual void syncModifiedCounter( const osgCompute::Context& context ) const;
		virtual bool allocPBO( TextureStream& stream ) const;
		virtual void syncPBO( TextureStream& stream ) const;
		virtual void syncTexture( TextureStream& stream ) const;

		osg::ref_ptr<osgCuda::TextureRectangle> _texref;
		bool							 _isRenderTarget;
	private:
		// copy constructor and operator should not be called
		TextureRectangleBuffer( const TextureRectangleBuffer& , const osg::CopyOp& ) {}
		TextureRectangleBuffer& operator=(const TextureRectangleBuffer&) { return (*this); }
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	TextureRectangleBuffer::TextureRectangleBuffer()
		: osgCuda::TextureBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	TextureRectangleBuffer::~TextureRectangleBuffer()
	{
		clearLocal();

		// notify TextureRectangle that proxy is now
		// deleted
		_texref->_proxy = NULL;
		// reattach handles
		_texref->_handles = getHandles();
		// decrease reference count of texture reference
		_texref = NULL;
	}

	//------------------------------------------------------------------------------
	void TextureRectangleBuffer::clear()
	{
		clearLocal();
		osgCuda::TextureBuffer::clear();
	}

	void TextureRectangleBuffer::clear( const osgCompute::Context& context ) const
	{
		TextureBuffer::clear( context );
	}

	//------------------------------------------------------------------------------
	bool TextureRectangleBuffer::init()
	{
		if( !isClear() )
			return true;

		if( !_texref.valid() )
			return false;

		_isRenderTarget = _texref->getIsRenderTarget();

		return osgCuda::TextureBuffer::init();
	}

	//------------------------------------------------------------------------------
	bool TextureRectangleBuffer::getIsRenderTarget() const
	{
		return _isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void TextureRectangleBuffer::setIsRenderTarget( bool isRenderTarget )
	{
		_isRenderTarget = isRenderTarget;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void TextureRectangleBuffer::clearLocal()
	{
		_isRenderTarget = false;
	}

	//------------------------------------------------------------------------------
	void TextureRectangleBuffer::syncModifiedCounter( const osgCompute::Context& context ) const
	{
		if( !_texref->getImage() )
			return;

		const osg::State* state = context.getGraphicsContext()->getState();
		if( state == NULL )
			return;

		_texref->getModifiedCount(state->getContextID()) = _texref->getImage()->getModifiedCount();
	}

	//------------------------------------------------------------------------------
	void TextureRectangleBuffer::syncTexture( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return;

		osg::GLBufferObject::Extensions* bufferExt = osg::GLBufferObject::getExtensions( state->getContextID(),true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncTexture(): cannot find required extension for context \""<<state->getContextID()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncTexture(): texture object not allocated for context \""<<state->getContextID()<<"\"."
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
		glBindTexture( GL_TEXTURE_RECTANGLE, tex->_id );

		// UNPACK the PBO data
		glTexSubImage2D(
			GL_TEXTURE_RECTANGLE, 0, 0, 0,
			getDimension(0),
			getDimension(1),
			tex->_profile._internalFormat,
			texType,
			NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncTexture(): error during glTex(Sub)ImageXD() for context \""
				<< state->getContextID()<<"\". Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}


		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	}

	//------------------------------------------------------------------------------
	void TextureRectangleBuffer::syncPBO( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return;

		osg::GLBufferObject::Extensions* bufferExt = osg::GLBufferObject::getExtensions( state->getContextID(),true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncPBO(): cannot find required extension for context \""<<state->getContextID()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncPBO() : texture object not allocated for context \""<<state->getContextID()<<"\"."
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
				<< "osgCuda::TextureRectangleBuffer::syncPBO(): texture type unknown."
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
				<< "osgCuda::TextureRectangleBuffer::syncPBO(): error during cudaGLUnregisterBufferObject() for context \""
				<< state->getContextID()<<"\"." << " " << cudaGetErrorString( res ) <<"."
				<< std::endl;

			return;
		}

		///////////////
		// COPY DATA //
		///////////////
		glBindTexture( GL_TEXTURE_RECTANGLE, tex->_id );
		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB,  stream._bo );

		// PACK the data for the PBO
		glGetTexImage( GL_TEXTURE_RECTANGLE, 0, tex->_profile._internalFormat, texType, NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncPBO(): error during glGetTexImage() for context \""
				<< state->getContextID()<<"\". Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}

		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, 0 );
		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

		////////////////////////
		// REGISTER PBO AGAIN //
		////////////////////////
		res = cudaGLRegisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::syncPBO(): error during cudaGLRegisterBufferObject() for context \""
				<< state->getContextID()<<"\"." << " " << cudaGetErrorString( res ) << "."
				<< std::endl;
		}

		stream._syncHost = true;
	}

	//------------------------------------------------------------------------------
	bool TextureRectangleBuffer::allocPBO( TextureStream& stream ) const
	{
		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return false;

		/////////////////////
		// COMPILE TEXTURE //
		/////////////////////
		osg::GLBufferObject::Extensions* bufferExt = osg::GLBufferObject::getExtensions( state->getContextID(), true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): cannot find required extension."
				<< std::endl;

			return false;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			_texref->apply( *state );
			glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
			GLenum errorStatus = glGetError();
			if( errorStatus != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureRectangleBuffer::allocPBO(): apply() failed on texture resource. Maybe context is not active."
					<< std::endl;

				return false;
			}

			// second chance
			tex = asTexture()->getTextureObject( state->getContextID() );
		}

		if( !tex )
		{

			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): cannot allocate texture object. Maybe context is not active."
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
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): texture type unknown."
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
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): cannot generate BufferObject (glGenBuffers())."
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
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): cannot bind BufferObject (glBindBuffer())."
				<< std::endl;

			return UINT_MAX;
		}

		bufferExt->glBufferData( GL_ARRAY_BUFFER_ARB, getByteSize(), tmpData, GL_DYNAMIC_DRAW );
		errorNo = glGetError();
		if (errorNo != GL_NO_ERROR)
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): cannot initialize BufferObject (glBufferData())."
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
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): something goes wrong within cudaGLRegisterBufferObject()."
				<< cudaGetErrorString(res) << "."
				<< std::endl;

			return UINT_MAX;
		}

		if( UINT_MAX == stream._bo )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangleBuffer::allocPBO(): Could not generate buffer object (glGenBuffers())."
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
			// Sync PBO with Texture-Data if Texture is allocated
			syncPBO( stream );
		}
		else if( !this->getIsRenderTarget() )
		{
			/////////////////////////////
			// ALLOCATE TEXTURE MEMORY //
			/////////////////////////////
			// else allocate the memory.
			glBindTexture( GL_TEXTURE_RECTANGLE, tex->_id );

			// Allocate memory for texture if not done so far in order to allow slot
			// to call glTexSubImage() during runtime
			glTexImage2D(
				GL_TEXTURE_RECTANGLE, 0,
				tex->_profile._internalFormat,
				tex->_profile._width, tex->_profile._height,
				tex->_profile._border,
				tex->_profile._internalFormat, texType, NULL );

			GLenum errorNo = glGetError();
			if( errorNo != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::TextureRectangleBuffer::allocPBO(): error during glTexImage2D(). Returned code is "
					<< std::hex<<errorNo<<"."
					<< std::endl;

				glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
				bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
				return false;
			}

			glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

			// Mark context based Texture-Object as allocated
			tex->setAllocated( true );
		}

		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	TextureRectangle::TextureRectangle()
		: osg::TextureRectangle(),
		  _proxy(NULL)
	{
		// some flags for textures are not available right now
		// like resize to a power of two and mipmaps
		asTexture()->setResizeNonPowerOfTwoHint( false );
		asTexture()->setUseHardwareMipMapGeneration( false );
	}

	//------------------------------------------------------------------------------
	TextureRectangle::~TextureRectangle()
	{
		if( _proxy != NULL )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::TextureRectangle::destructor(): proxy is still valid!!!."
				<< std::endl;
		}
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* TextureRectangle::getBuffer()
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	const osgCompute::InteropBuffer* TextureRectangle::getBuffer() const
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* TextureRectangle::getOrCreateBuffer()
	{
		// create proxy buffer on demand
		if( _proxy == NULL )
		{
			_proxy = new TextureRectangleBuffer;
			_proxy->setHandles( _handles );
			_handles.clear();
			_proxy->_texref = this;
			if( !_proxy->init() )
			{
				_proxy->unref();
				_proxy = NULL;
			}
		}

		return _proxy;
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::addHandle( const std::string& handle )
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
	void TextureRectangle::removeHandle( const std::string& handle )
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
	bool TextureRectangle::isAddressedByHandle( const std::string& handle ) const
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
	void TextureRectangle::releaseGLObjects( osg::State* state/*=0*/ ) const
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

		osg::TextureRectangle::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::apply( osg::State& state ) const
	{
		const osgCompute::Context* curCtx = osgCompute::Context::getContext( state.getContextID() );
		if( curCtx )
		{
			if( NULL != _proxy && _proxy->getMapping( *curCtx ) != osgCompute::UNMAPPED )
				_proxy->unmap( *curCtx );
		}

		osg::TextureRectangle::apply( state );
	}

	//------------------------------------------------------------------------------
	bool TextureRectangle::getIsRenderTarget() const
	{
		return _isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::setIsRenderTarget( bool isRenderTarget )
	{
		_isRenderTarget = isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::clear()
	{
		clearLocal();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void TextureRectangle::clearLocal()
	{
		_isRenderTarget = false;
		if( NULL != _proxy )
			_proxy->clear();

		_handles.clear();
	}
}
