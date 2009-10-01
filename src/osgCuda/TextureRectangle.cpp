#include <osg/GL>
#include <osg/Texture>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/TextureRectangle>

namespace osgCuda
{
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	TextureRectangle::TextureRectangle()
		:	osgCuda::Texture(),
			osg::TextureRectangle()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	TextureRectangle::~TextureRectangle()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::clear()
	{
		osgCuda::Texture::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool TextureRectangle::init()
	{
		return osgCuda::Texture::init();
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::releaseGLObjects( osg::State* state/*=0*/ ) const
	{
		if( state != NULL )
		{
			const osgCompute::Context* curCtx = this->getContext( state->getContextID() );
			if( curCtx )
			{
				if( getMapping( *curCtx ) != osgCompute::UNMAPPED )
					this->unmap( *curCtx );
			}
		}

		osg::TextureRectangle::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::apply( osg::State& state ) const
	{
		const osgCompute::Context* curCtx = this->getContext( state.getContextID() );
		if( curCtx )
		{
			if( getMapping( *curCtx ) != osgCompute::UNMAPPED )
				this->unmap( *curCtx );
		}

		osg::TextureRectangle::apply( state );
	}

	//------------------------------------------------------------------------------
	bool TextureRectangle::initDimension()
	{
		if( getNumDimensions() != 0 )
		{
			setTextureWidth( getDimension(0) );
			setTextureHeight( getDimension(1) );
		}
		else
		{
			if( getImage(0) && getTextureWidth() == 0)
				setTextureWidth( getImage(0)->s() );

			if( getImage(0) && getTextureHeight() == 0)
				setTextureHeight( getImage(0)->t() );

			setDimension(0, getTextureWidth() );
			setDimension(1, getTextureHeight() );
		}

		return osgCuda::Texture::initDimension();
	}

	//------------------------------------------------------------------------------
	osg::Image* TextureRectangle::getImagePtr()
	{
		return this->getImage();
	}

	//------------------------------------------------------------------------------
	const osg::Image* TextureRectangle::getImagePtr() const
	{
		return this->getImage();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void TextureRectangle::clearLocal()
	{
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::syncModifiedCounter( const osgCompute::Context& context ) const
	{
		if( !getImagePtr() )
			return; 

		osg::TextureRectangle::getModifiedCount(context.getState()->getContextID()) = getImagePtr()->getModifiedCount();
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::syncTexture( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( stream._context->getId(),true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncTexture() for texture \""
				<< osg::Object::getName()<< "\": cannot find required extension for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncTexture() for texture \""
				<< osg::Object::getName()<< "\": texture object not allocated for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return;
		}

		GLenum texType = GL_NONE;
		if( osg::Texture::getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT )
		{
			if( !getImage() )
				return;

			texType = getImage()->getDataType();
		}
		else
		{
			texType = osg::Texture::getSourceType();
		}

		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB,  stream._bo );
		glBindTexture( GL_TEXTURE_RECTANGLE, tex->_id );

		// UNPACK the PBO data
		glTexSubImage2D(
			GL_TEXTURE_RECTANGLE, 0, 0, 0,
			getDimension(0),
			getDimension(1),
			tex->_internalFormat,
			texType,
			NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncTexture() for buffer \""
				<< osg::Object::getName()<< "\": error during glTex(Sub)ImageXD() for context \""
				<< stream._context->getId()<<"\". Returned code is "
				<< std::hex<<errorStatus<<"."
				<< std::endl;
		}


		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	}

	//------------------------------------------------------------------------------
	void TextureRectangle::syncPBO( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( stream._context->getId(),true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": cannot find required extension for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": texture object not allocated for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return;
		}

		GLenum texType = GL_NONE;
		if( osg::Texture::getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT &&
			getImage() )
		{
			texType = getImage()->getDataType();
		}
		else
		{
			texType = osg::Texture::getSourceType();
		}

		if( texType == GL_NONE )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": texture type unknown."
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
				<< "CUDA::TextureRectangle::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": error during cudaGLUnregisterBufferObject() for context \""
				<< stream._context->getId()<<"\". Returned code is "
				<< std::hex<<res<<"."
				<< std::endl;

			return;
		}

		///////////////
		// COPY DATA //
		///////////////
		glBindTexture( GL_TEXTURE_RECTANGLE, tex->_id );
		bufferExt->glBindBuffer( GL_PIXEL_PACK_BUFFER_ARB,  stream._bo );

		// PACK the data for the PBO
		glGetTexImage( GL_TEXTURE_RECTANGLE, 0, tex->_internalFormat, texType, NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": error during glGetTexImage() for context \""
				<< stream._context->getId()<<"\". Returned code is "
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
				<< "CUDA::TextureRectangle::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": error during cudaGLRegisterBufferObject() for context \""
				<< stream._context->getId()<<"\". Returned code is "
				<< std::hex<<res<<"."
				<< std::endl;
		}

		stream._syncHost = true;
	}

	//------------------------------------------------------------------------------
	bool TextureRectangle::allocPBO( TextureStream& stream ) const
	{
		/////////////////////
		// COMPILE TEXTURE //
		/////////////////////
		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( stream._context->getId(), true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::allocPBO() for texture \""
				<< osg::Object::getName()<< "\": cannot find required extension for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return false;
		}

		osg::Texture::TextureObject* tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{
			apply( *stream._context->getState() );
			glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
		}

		// second chance
		tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{

			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::allocPBO() for texture \""
				<< osg::Object::getName()<< "\": texture object not allocated for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return false;
		}

		GLenum texType = GL_NONE;
		if( osg::Texture::getInternalFormatMode() == osg::Texture::USE_IMAGE_DATA_FORMAT &&
			getImage() )
		{
			texType = getImage()->getDataType();
		}
		else
		{
			texType = osg::Texture::getSourceType();
		}

		if( texType == GL_NONE )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::allocPBO() for texture \""
				<< osg::Object::getName()<< "\": texture type unknown."
				<< std::endl;

			return false;
		}

		///////////////
		// SETUP PBO //
		///////////////
		stream._bo = static_cast<const Context*>( stream._context.get() )->mallocBufferObject( getByteSize() );
		if( UINT_MAX == stream._bo )
		{
			osg::notify(osg::FATAL)
				<< "CUDA::TextureRectangle::allocPBO() for Buffer \""
				<< osg::Object::getName()<< "\": Could not generate buffer object (glGenBuffers()) for context \""
				<< stream._context->getId()<<"\"."
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
			// Sync PBO with texture data if texture is allocated
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
				tex->_internalFormat,
				tex->_width, tex->_height,
				tex->_border,
				tex->_internalFormat, texType, NULL );

			GLenum errorNo = glGetError();
			if( errorNo != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "CUDA::TextureRectangle::allocPBO() for buffer \""
					<< osg::Object::getName()<< "\": error during glTexImageXD() for context \""
					<< stream._context->getId()<<"\". Returned code is "
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

	//------------------------------------------------------------------------------
	osgCompute::BufferStream* TextureRectangle::newStream( const osgCompute::Context& context ) const
	{
		return new TextureStream;
	}
}