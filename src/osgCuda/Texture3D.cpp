#include <osg/GL>
#include <osg/Texture>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/Texture3D>

namespace osgCuda
{

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Texture3D::Texture3D()
		: osgCuda::Texture(),
          osg::Texture3D()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	Texture3D::~Texture3D()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void Texture3D::clear()
	{
		Texture::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool Texture3D::init()
	{
		return Texture::init();
	}

	//------------------------------------------------------------------------------
	void Texture3D::releaseGLObjects( osg::State* state/*=0*/ ) const
	{
		if( state != NULL )
		{
			const osgCompute::Context* curCtx = getContext( state->getContextID() );
			if( curCtx )
			{
				if( osgCompute::Buffer::getMapping( *curCtx ) != osgCompute::UNMAPPED )
					unmap( *curCtx );
			}
		}

		osg::Texture3D::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void Texture3D::apply( osg::State& state ) const
	{
		const osgCompute::Context* curCtx = getContext( state.getContextID() );
		if( curCtx )
		{
			if( osgCompute::Buffer::getMapping( *curCtx ) != osgCompute::UNMAPPED )
				unmap( *curCtx );
		}

		osg::Texture3D::apply( state );
	}

	//------------------------------------------------------------------------------
	bool Texture3D::initDimension()
	{
		if( osgCompute::Buffer::getNumDimensions() != 0 )
		{
			setTextureWidth( osgCompute::Buffer::getDimension(0) );
			setTextureHeight( osgCompute::Buffer::getDimension(1) );
			setTextureDepth( osgCompute::Buffer::getDimension(2) );
		}
		else
		{
			if( getImage(0) && getTextureWidth() == 0)
				setTextureWidth( getImage(0)->s() );

			if( getImage(0) && getTextureHeight() == 0)
				setTextureHeight( getImage(0)->t() );

			if( getImage(0) && getTextureDepth() == 0)
				setTextureDepth( getImage(0)->r() );

			osgCompute::Buffer::setDimension(0, getTextureWidth() );
			osgCompute::Buffer::setDimension(1, getTextureHeight() );
			osgCompute::Buffer::setDimension(2, getTextureDepth() );
		}

		return Texture::initDimension();
	}

	//------------------------------------------------------------------------------
	osg::Image* Texture3D::getImagePtr()
	{
		return this->getImage();
	}

	//------------------------------------------------------------------------------
	const osg::Image* Texture3D::getImagePtr() const
	{
		return this->getImage();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	void Texture3D::clearLocal()
	{
	}

	//------------------------------------------------------------------------------
	void Texture3D::syncModifiedCounter( const osgCompute::Context& context ) const
	{
		if( !getImagePtr() )
			return;

		osg::Texture3D::getModifiedCount(context.getState()->getContextID()) = getImagePtr()->getModifiedCount();
	}

	//------------------------------------------------------------------------------
	void Texture3D::syncTexture( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( stream._context->getId(), true );
		osg::Texture3D::Extensions* tex3DExt = osg::Texture3D::getExtensions( stream._context->getId(), true );
		if( !bufferExt || !tex3DExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::syncTexture() for texture \""
				<< osg::Object::getName()<< "\": cannot find required extensions for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::syncTexture() for texture \""
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
		glBindTexture( GL_TEXTURE_3D, tex->_id );

		// UNPACK the PBO data
		tex3DExt->glTexSubImage3D(
			GL_TEXTURE_3D, 0, 0, 0, 0,
			osgCompute::Buffer::getDimension(0),
			osgCompute::Buffer::getDimension(1),
			osgCompute::Buffer::getDimension(2),
			tex->_internalFormat,
			texType,
			NULL );

		GLenum errorStatus = glGetError();
		if( errorStatus != GL_NO_ERROR )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::syncTexture() for buffer \""
				<< osg::Object::getName()<< "\": error during glTex(Sub)ImageXD() for context \""
				<< stream._context->getId()<<"\". Returned code is "
				<< std::hex<<errorStatus<< "."
				<< std::endl;
		}


		glBindTexture( GL_TEXTURE_3D, 0 );
		bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	}

	//------------------------------------------------------------------------------
	void Texture3D::syncPBO( TextureStream& stream ) const
	{
		if( stream._bo == UINT_MAX )
			return;

		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( stream._context->getId(),true );
		if( !bufferExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": cannot find required extension for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return;
		}

		osg::Texture::TextureObject* tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::syncPBO() for texture \""
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
				<< "osgCuda::Texture3D::syncPBO() for texture \""
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
				<< "osgCuda::Texture3D::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": error during cudaGLUnregisterBufferObject() for context \""
				<< stream._context->getId()<<"\"."
				<< " " << cudaGetErrorString( res ) <<"."
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
				<< "osgCuda::Texture3D::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": error during glGetTexImage() for context \""
				<< stream._context->getId()<<"\". Returned code is "
				<< std::hex<<errorStatus<< "."
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
				<< "osgCuda::Texture3D::syncPBO() for texture \""
				<< osg::Object::getName()<< "\": error during cudaGLRegisterBufferObject() for context \""
				<< stream._context->getId()<<"\"."
				<< " " << cudaGetErrorString( res ) <<"."
				<< std::endl;
		}

		stream._syncHost = true;
	}

	//------------------------------------------------------------------------------
	bool Texture3D::allocPBO( TextureStream& stream ) const
	{
		/////////////////////
		// COMPILE TEXTURE //
		/////////////////////
		osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( stream._context->getId(), true );
		osg::Texture3D::Extensions* tex3DExt = osg::Texture3D::getExtensions( stream._context->getId(),true );
		if( !bufferExt || !tex3DExt )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::allocPBO() for texture \""
				<< osg::Object::getName()<< "\": cannot find required extensions for context \""<<stream._context->getId()<<"\"."
				<< std::endl;

			return false;
		}

		osg::Texture::TextureObject* tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{
			apply( *stream._context->getState() );
			glBindTexture( GL_TEXTURE_3D, 0 );
			GLenum errorStatus = glGetError();
			if( errorStatus != GL_NO_ERROR )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Texture3D::allocPBO() for buffer \""
					<< osg::Object::getName()<<"\": apply() failed on texture resource."
					<< std::endl;

				return false;
			}
		}

		// second chance
		tex = osg::Texture::getTextureObject( stream._context->getId() );
		if( !tex )
		{

			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::allocPBO() for texture \""
				<< osg::Object::getName()<< "\":cannot allocate texture object for context \""<<stream._context->getId()<<"\"."
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
				<< "osgCuda::Texture3D::allocPBO() for texture \""
				<< osg::Object::getName()<< "\": texture type unknown."
				<< std::endl;

			return false;
		}

		///////////////
		// SETUP PBO //
		///////////////
		stream._bo = static_cast<const Context*>( stream._context.get() )->mallocBufferObject( osgCompute::Buffer::getByteSize() );
		if( UINT_MAX == stream._bo )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Texture3D::allocPBO() for Buffer \""
				<< osg::Object::getName()<< "\": Could not generate buffer object (glGenBuffers()) for context \""
				<< stream._context->getId()<<"\"."
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
			// Sync PBO with Texture-Data if Texture is allocated
			syncPBO( stream );
		}
		else if( !getIsRenderTarget() )
		{
			///////////////////////////
			// ALLOCATE CLEAN MEMORY //
			///////////////////////////
			// else allocate the memory.
			glBindTexture( GL_TEXTURE_3D, tex->_id );

			// Allocate memory for texture if not done so far in order to allow slot
			// to call glTexSubImage() during runtime
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
					<< "osgCuda::Texture3D::allocPBO() for buffer \""
					<< osg::Object::getName()<< "\": error during glTexImageXD() for context \""
					<< stream._context->getId()<<"\". Returned code is "
					<< std::hex<<errorNo<<"."
					<< std::endl;

				glBindTexture( GL_TEXTURE_3D, 0 );
				bufferExt->glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
				return false;
			}

			glBindTexture( GL_TEXTURE_3D, 0 );

			// Mark context based Texture-Object as allocated
			tex->setAllocated( true );
		}

		return true;
	}

	//------------------------------------------------------------------------------
	osgCompute::BufferStream* Texture3D::newStream( const osgCompute::Context& context ) const
	{
		return new TextureStream;
	}
}
