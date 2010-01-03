#if defined(__linux)
    #include <malloc.h>
#endif
#include <memory.h>
#include <osg/GL>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/Texture>

#include <GL/GLU.h>

namespace osgCuda
{
	/**
	*/
	class Texture2DBuffer : public osgCuda::TextureBuffer
	{
	public:
		Texture2DBuffer();

		META_Object(osgCuda,Texture2DBuffer)

		virtual bool init();

		virtual osgCompute::InteropObject* getObject() { return _texref.get(); }
		virtual osg::Texture* asTexture() { return _texref.get(); }
		virtual const osg::Texture* asTexture() const { return _texref.get(); }

		virtual bool getIsRenderTarget() const;

		virtual void clear();
	protected:
		friend class Texture2D;
		virtual ~Texture2DBuffer();
		void clearLocal();

		virtual void setIsRenderTarget( bool isRenderTarget );
		virtual void syncModifiedCounter( const osgCompute::Context& context ) const;
		virtual bool allocPBO( TextureStream& stream );
		virtual void syncPBO( TextureStream& stream );
		virtual void syncTexture( TextureStream& stream );

		virtual void clear( const osgCompute::Context& context ) const;

		osg::ref_ptr<osgCuda::Texture2D> _texref;
		bool							 _isRenderTarget;
	private:
		// copy constructor and operator should not be called
		Texture2DBuffer( const Texture2DBuffer& , const osg::CopyOp& ) {}
		Texture2DBuffer& operator=(const Texture2DBuffer&) { return (*this); }
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture2DBuffer::Texture2DBuffer()
		: osgCuda::TextureBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	Texture2DBuffer::~Texture2DBuffer()
	{
		clearLocal();

		// notify Texture2D that proxy is now
		// deleted
		_texref->_proxy = NULL;
		// reattach handles
		_texref->_handles = getHandles();
		// decrease reference count of texture reference
		_texref = NULL;
	}

	//------------------------------------------------------------------------------
	void Texture2DBuffer::clear()
	{
		osgCuda::TextureBuffer::clear();
		clearLocal();
	}

	void Texture2DBuffer::clear( const osgCompute::Context& context ) const
	{
		TextureBuffer::clear( context );
	}

	//------------------------------------------------------------------------------
	bool Texture2DBuffer::init()
	{
		if( !isClear() )
			return true;

		if( !_texref.valid() )
			return false;

		_isRenderTarget = _texref->getIsRenderTarget();

		return osgCuda::TextureBuffer::init();
	}

	//------------------------------------------------------------------------------
	bool Texture2DBuffer::getIsRenderTarget() const
	{
		return _isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void Texture2DBuffer::setIsRenderTarget( bool isRenderTarget )
	{
		_isRenderTarget = isRenderTarget;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture2DBuffer::clearLocal()
	{
		_isRenderTarget = false;
	}

	//------------------------------------------------------------------------------
	void Texture2DBuffer::syncModifiedCounter( const osgCompute::Context& context ) const
	{
		if( !_texref->getImage() )
			return;

		const osg::State* state = context.getGraphicsContext()->getState();
		if( state == NULL )
			return;

		_texref->getModifiedCount(state->getContextID()) = _texref->getImage()->getModifiedCount();
	}

	//------------------------------------------------------------------------------
	void Texture2DBuffer::syncTexture( TextureStream& stream )
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
				<< getName() << " [osgCuda::Texture2DBuffer::syncTexture()]: cannot find required extension."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::syncTexture()]: texture object not allocated."
				<< std::endl;

			return;
		}

		GLenum format = osg::Image::computePixelFormat( asTexture()->getInternalFormat() );
		GLenum type = osg::Image::computeFormatDataType( asTexture()->getInternalFormat() );

		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB,  stream._bo );
		glBindTexture( GL_TEXTURE_2D, tex->_id );

		// UNPACK the PBO data
		glTexSubImage2D(
			GL_TEXTURE_2D, 0, 0, 0,
			getDimension(0),
			getDimension(1),
			format,
			type,
			0 );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::syncTexture()]: error during glTex(Sub)ImageXD(). Returned code is "
				<< std::hex<<errorStatus
				<< std::endl;

			
		}

		glBindTexture( GL_TEXTURE_2D, 0 );
		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	}

	//------------------------------------------------------------------------------
	void Texture2DBuffer::syncPBO( TextureStream& stream )
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
				<< getName() << " [osgCuda::Texture2DBuffer::syncPBO()]: cannot find required extension."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::syncPBO()]: texture object not allocated ."
				<< std::endl;

			return;
		}

		GLenum format = osg::Image::computePixelFormat( asTexture()->getInternalFormat() );
		GLenum type = osg::Image::computeFormatDataType( asTexture()->getInternalFormat() );

		////////////////////
		// UNREGISTER PBO //
		////////////////////
		cudaError res = cudaGLUnregisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::syncPBO()]: error during cudaGLUnregisterBufferObject()." 
				<< " " << cudaGetErrorString( res ) <<"."
				<< std::endl;

			return;
		}

		///////////////
		// COPY DATA //
		///////////////
		glBindTexture( GL_TEXTURE_2D, tex->_id );
		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB,  stream._bo );

		// PACK the data for the PBO
		glGetTexImage( GL_TEXTURE_2D, 0, format, type, NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::syncPBO()]: error during glGetTexImage() for context \""
				<< state->getContextID()<<"\". Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}

		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );

		////////////////////////
		// REGISTER PBO AGAIN //
		////////////////////////
		res = cudaGLRegisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::syncPBO()]: error during cudaGLRegisterBufferObject()."
				<< std::endl;
		}

		stream._syncHost = true;
	}

	//------------------------------------------------------------------------------
	bool Texture2DBuffer::allocPBO( TextureStream& stream )
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
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: cannot find required extension."
				<< std::endl;

			return false;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			_texref->apply( *state );
			glBindTexture( GL_TEXTURE_2D, 0 );
			GLenum errorStatus = glGetError();
			if( errorStatus != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: apply() failed on texture resource. Maybe context is not active."
					<< std::endl;

				return false;
			}

			// second chance
			tex = asTexture()->getTextureObject( state->getContextID() );
		}

		if( !tex )
		{

			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: cannot allocate texture object. Maybe context is not active."
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
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: cannot generate BufferObject (glGenBuffers())."
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
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: cannot bind BufferObject (glBindBuffer())."
				<< std::endl;

			return UINT_MAX;
		}

		bufferExt->glBufferData( GL_ARRAY_BUFFER_ARB, getByteSize(), tmpData, GL_STATIC_DRAW );
		errorNo = glGetError();
		if (errorNo != GL_NO_ERROR)
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: cannot initialize BufferObject (glBufferData())."
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
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: something goes wrong within cudaGLRegisterBufferObject()."
				<< cudaGetErrorString(res) << "."
				<< std::endl;

			return UINT_MAX;
		}

		if( UINT_MAX == stream._bo )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2DBuffer::allocPBO()]: cannot generate buffer object (glGenBuffers())."
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

		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture2D::Texture2D()
		: osg::Texture2D(),
		  _proxy(NULL)
	{
		clearLocal();

		// some flags for textures are not available right now
		// like resize to a power of two and mipmaps
		asTexture()->setResizeNonPowerOfTwoHint( false );
		asTexture()->setUseHardwareMipMapGeneration( false );
	}

	//------------------------------------------------------------------------------
	Texture2D::~Texture2D()
	{
		if( _proxy != NULL )
		{
			osg::notify(osg::FATAL)
				<< getName() << " [osgCuda::Texture2D::destructor()]: proxy is still valid!!!."
				<< std::endl;
		}

		clearLocal();
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Texture2D::getBuffer()
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	const osgCompute::InteropBuffer* Texture2D::getBuffer() const
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Texture2D::getOrCreateBuffer()
	{
		// create proxy buffer on demand
		if( _proxy == NULL )
		{
			_proxy = new Texture2DBuffer;
			_proxy->_texref = this;
			_proxy->setHandles( _handles );
			_handles.clear();
			if( !_proxy->init() )
			{
				_proxy->unref();
				_proxy = NULL;
			}
		}

		// print warning if the source format and source type do not match
		// the internal texture format
		GLenum format = osg::Image::computePixelFormat( asTexture()->getInternalFormat() );
		GLenum type = osg::Image::computeFormatDataType( asTexture()->getInternalFormat() );
		if( NULL == getImage() && 
			(format != getSourceFormat() || type != getSourceType()) )
		{
			osg::notify( osg::WARN ) 
				<< getName() << " [osgCuda::Texture2D::getOrCreateBuffer()]: No image found but type and format do not match the internal texture format."
				<< "This might lead to errors during glTexSubImage1D(). Please make sure to setup source type and format properly."
				<< std::endl;
		}

		return _proxy;
	}

	//------------------------------------------------------------------------------
	void Texture2D::addHandle( const std::string& handle )
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
	void Texture2D::removeHandle( const std::string& handle )
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
	bool Texture2D::isAddressedByHandle( const std::string& handle ) const
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
	void Texture2D::releaseGLObjects( osg::State* state/*=0*/ ) const
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

		osg::Texture2D::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void Texture2D::apply( osg::State& state ) const
	{
		const osgCompute::Context* curCtx = osgCompute::Context::getContext( state.getContextID() );
		if( curCtx && _proxy != NULL )
		{
			if( osgCompute::Context::getAppliedContext() != curCtx )
				const_cast<osgCompute::Context*>(curCtx)->apply();

			_proxy->checkMappingWithinApply( *curCtx );
		}

		osg::Texture2D::apply( state );
	}

	//------------------------------------------------------------------------------
	bool Texture2D::getIsRenderTarget() const
	{
		return _isRenderTarget;
	}

	//------------------------------------------------------------------------------
	void Texture2D::setIsRenderTarget( bool isRenderTarget )
	{
		_isRenderTarget = isRenderTarget;
	}
	//------------------------------------------------------------------------------
	void Texture2D::clear()
	{
		clearLocal();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture2D::clearLocal()
	{
		_isRenderTarget = false;
		if( NULL != _proxy )
			_proxy->clear();

		_handles.clear();
	}
}
