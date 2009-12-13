#include <memory.h>
#include <malloc.h>
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
	class Texture1DBuffer : public osgCuda::TextureBuffer
	{
	public:
		Texture1DBuffer();

		META_Object(osgCuda,Texture1DBuffer)

		virtual bool init();

		virtual osgCompute::InteropObject* getObject() { return _texref.get(); }
		virtual osg::Texture* asTexture() { return _texref.get(); }
		virtual const osg::Texture* asTexture() const { return _texref.get(); }

		virtual bool getIsRenderTarget() const;

		virtual void clear( const osgCompute::Context& context ) const;
		virtual void clear();
	protected:
		friend class Texture1D;
		virtual ~Texture1DBuffer();
		void clearLocal();

		virtual void syncModifiedCounter( const osgCompute::Context& context ) const;
		virtual bool allocPBO( TextureStream& stream ) const;
		virtual void syncPBO( TextureStream& stream ) const;
		virtual void syncTexture( TextureStream& stream ) const;

		osg::ref_ptr<osgCuda::Texture1D> _texref;
		bool							 _isRenderTarget;
	private:
		// copy constructor and operator should not be called
		Texture1DBuffer( const Texture1DBuffer& , const osg::CopyOp& ) {}
		Texture1DBuffer& operator=(const Texture1DBuffer&) { return (*this); }
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture1DBuffer::Texture1DBuffer()
		: osgCuda::TextureBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	Texture1DBuffer::~Texture1DBuffer()
	{
		clearLocal();

		// notify Texture1D that proxy is now
		// deleted
		_texref->_proxy = NULL;
		// reattach handles
		_texref->_handles = getHandles();
		// decrease reference count of texture reference
		_texref = NULL;
	}

	//------------------------------------------------------------------------------
	void Texture1DBuffer::clear()
	{
		osgCuda::TextureBuffer::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void Texture1DBuffer::clear( const osgCompute::Context& context ) const
	{
		TextureBuffer::clear( context );
	}

	//------------------------------------------------------------------------------
	bool Texture1DBuffer::init()
	{
		if( !isClear() )
			return true;

		if( !_texref.valid() )
			return false;

		return osgCuda::TextureBuffer::init();
	}

	//------------------------------------------------------------------------------
	bool Texture1DBuffer::getIsRenderTarget() const
	{
		return false;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture1DBuffer::clearLocal()
	{
		_isRenderTarget = false;
	}

	//------------------------------------------------------------------------------
	void Texture1DBuffer::syncModifiedCounter( const osgCompute::Context& context ) const
	{
		if( !_texref->getImage() )
			return;

		const osg::State* state = context.getGraphicsContext()->getState();
		if( state == NULL )
			return;

		_texref->getModifiedCount(state->getContextID()) = _texref->getImage()->getModifiedCount();
	}

	//------------------------------------------------------------------------------
	void Texture1DBuffer::syncTexture( TextureStream& stream ) const
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
				<< "osgCuda::Texture1DBuffer::syncTexture(): cannot find required extension for context \""<<state->getContextID()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::syncTexture(): texture object is not allocated for context \""<<state->getContextID()<<"\"."
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
		glBindTexture( GL_TEXTURE_1D, tex->_id );

		// UNPACK the PBO data
		glTexSubImage1D(
			GL_TEXTURE_1D, 0, 0,
			getDimension(0),
			tex->_profile._internalFormat,
			texType,
			NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::syncTexture(): error during glTex(Sub)ImageXD() for context \""
				<< state->getContextID()<<"\". Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}


		glBindTexture( GL_TEXTURE_1D, 0 );
		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	}

	//------------------------------------------------------------------------------
	void Texture1DBuffer::syncPBO( TextureStream& stream ) const
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
				<< "osgCuda::Texture1DBuffer::syncPBO(): cannot find required extension for context \""<<state->getContextID()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::syncPBO() : texture object not allocated for context \""<<state->getContextID()<<"\"."
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
				<< "osgCuda::Texture1DBuffer::syncPBO(): texture type unknown."
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
				<< "osgCuda::Texture1DBuffer::syncPBO(): error during cudaGLUnregisterBufferObject() for context \""
				<< state->getContextID()<<"\"." << " " << cudaGetErrorString( res ) <<"."
				<< std::endl;

			return;
		}

		///////////////
		// COPY DATA //
		///////////////
		glBindTexture( GL_TEXTURE_1D, tex->_id );
		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB,  stream._bo );

		// PACK the data for the PBO
		glGetTexImage( GL_TEXTURE_1D, 0, tex->_profile._internalFormat, texType, NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::syncPBO(): error during glGetTexImage() for context \""
				<< state->getContextID()<<"\". Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}

		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB, 0 );
		glBindTexture( GL_TEXTURE_1D, 0 );

		////////////////////////
		// REGISTER PBO AGAIN //
		////////////////////////
		res = cudaGLRegisterBufferObject( stream._bo );
		if( cudaSuccess != res )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::syncPBO(): error during cudaGLRegisterBufferObject() for context \""
				<< state->getContextID()<<"\"." << " " << cudaGetErrorString( res ) << "."
				<< std::endl;
		}

		stream._syncHost = true;
	}

	//------------------------------------------------------------------------------
	bool Texture1DBuffer::allocPBO( TextureStream& stream ) const
	{
		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return false;

		/////////////////////
		// COMPILE TEXTURE //
		/////////////////////
		osg::GLBufferObject::Extensions* bufferExt = osg::GLBufferObject::getExtensions( state->getContextID(), true );
		if( !bufferExt  )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::allocPBO(): cannot find required extensions for context \""<<state->getContextID()<<"\"."
				<< std::endl;

			return false;
		}

		osg::Texture::TextureObject* tex = asTexture()->getTextureObject( state->getContextID() );
		if( !tex )
		{
			asTexture()->apply( *state );
			glBindTexture( GL_TEXTURE_1D, 0 );
			GLenum errorStatus = glGetError();
			if( errorStatus != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture1DBuffer::allocPBO(): apply() failed on texture resource. Maybe context is not active."
					<< std::endl;

				return false;
			}


			// second chance
			tex = asTexture()->getTextureObject( state->getContextID() );
		}

		if( !tex )
		{

			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::allocPBO(): cannot allocate texture object. Maybe context is not active."
				<< std::endl;

			return false;
		}

		GLenum texType = GL_NONE;
		if( asTexture()->getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT &&
			asTexture()->getImage(0) )
		{
			texType = asTexture()->getImage(0)->getDataType();
		}
		else
		{
			texType = asTexture()->getSourceType();
		}

		if( texType == GL_NONE )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::allocPBO(): texture type unknown."
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
				<< "osgCuda::Texture1DBuffer::allocPBO(): cannot generate BufferObject (glGenBuffers())."
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
				<< "osgCuda::Texture1DBuffer::allocPBO(): cannot bind BufferObject (glBindBuffer())."
				<< std::endl;

			return UINT_MAX;
		}

		bufferExt->glBufferData( GL_ARRAY_BUFFER_ARB, getByteSize(), tmpData, GL_DYNAMIC_DRAW );
		errorNo = glGetError();
		if (errorNo != GL_NO_ERROR)
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::allocPBO(): cannot initialize BufferObject (glBufferData())."
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
				<< "osgCuda::Texture1DBuffer::allocPBO(): something goes wrong within cudaGLRegisterBufferObject()."
				<< cudaGetErrorString(res) << "."
				<< std::endl;

			return UINT_MAX;
		}

		if( UINT_MAX == stream._bo )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1DBuffer::allocPBO(): Could not generate buffer object (glGenBuffers()) for context \""
				<< state->getContextID()<<"\"."
				<< std::endl;

			return false;
		}
		stream._boRegistered = true;

		////////////////
		// SETUP DATA //
		////////////////
		if( tex->isAllocated() )
		{
			//////////////
			// SYNC PBO //
			//////////////
			// Sync PBO with texture-data
			// if it is already allocated
			syncPBO( stream );
		}

		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture1D::Texture1D()
		: osg::Texture1D(),
		  _proxy(NULL)
	{
		clearLocal();
		// some flags for textures are not available right now
		// like resize to a power of two and mip-maps
		asTexture()->setResizeNonPowerOfTwoHint( false );
		asTexture()->setUseHardwareMipMapGeneration( false );
	}

	//------------------------------------------------------------------------------
	Texture1D::~Texture1D()
	{
		if( _proxy != NULL )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture1D::destructor(): proxy is still valid!!!."
				<< std::endl;
		}

		clearLocal();
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Texture1D::getBuffer()
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	const osgCompute::InteropBuffer* Texture1D::getBuffer() const
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Texture1D::getOrCreateBuffer()
	{
		// create proxy buffer on demand
		if( _proxy == NULL )
		{
			_proxy = new Texture1DBuffer;
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
	void Texture1D::addHandle( const std::string& handle )
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
	void Texture1D::removeHandle( const std::string& handle )
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
	bool Texture1D::isAddressedByHandle( const std::string& handle ) const
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
	void Texture1D::releaseGLObjects( osg::State* state/*=0*/ ) const
	{
		if( state != NULL )
		{
			const osgCompute::Context* curCtx = osgCompute::Context::getContextFromGraphicsContext( state->getContextID() );
			if( curCtx )
			{
				if( NULL != _proxy )
					_proxy->clear( *curCtx );
			}
		}

		osg::Texture1D::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void Texture1D::apply( osg::State& state ) const
	{
		const osgCompute::Context* curCtx = osgCompute::Context::getContextFromGraphicsContext( state.getContextID() );
		if( curCtx && _proxy != NULL )
			_proxy->checkMappingWithinApply( *curCtx );

		osg::Texture1D::apply( state );
	}

	//------------------------------------------------------------------------------
	void Texture1D::clear()
	{
		clearLocal();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture1D::clearLocal()
	{
		if( NULL != _proxy )
			_proxy->clear();

		_handles.clear();
	}
}
