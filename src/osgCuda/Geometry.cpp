#include <memory.h>
#include <osg/GL>
#include <osg/RenderInfo>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/Geometry>

namespace osgCuda
{
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	GeometryStream::GeometryStream()
		:	osgCompute::BufferStream(),
			_devPtr(NULL),
			_hostPtr(NULL),
			_syncDevice(false),
			_syncHost(false),
			_hostPtrAllocated(false),
			_bo( UINT_MAX ),
			_boRegistered(false)
	{
	}

	//------------------------------------------------------------------------------
	GeometryStream::~GeometryStream()
	{
		if( _boRegistered && _bo != UINT_MAX )
			static_cast<Context*>( osgCompute::BufferStream::_context.get() )->freeBufferObject( _bo );
		if( _hostPtrAllocated && NULL != _hostPtr)
			static_cast<Context*>(osgCompute::BufferStream::_context.get())->freeMemory( _hostPtr );
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Geometry::Geometry()
		:   osgCompute::Buffer(),
			osg::Geometry()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	Geometry::~Geometry()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void Geometry::clear()
	{
		osgCompute::Buffer::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	bool Geometry::init()
	{
		if( !osgCompute::Resource::isClear() )
			return true;

		// geometry must use vertex buffer objects
		setUseVertexBufferObjects( true );

		if( osg::Geometry::getVertexArray() == NULL || osg::Geometry::getVertexArray()->getNumElements() == 0 )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Geometry::initDimension() for geometry \""
				<< osg::Object::getName()<< "\": no dimensions defined for geometry! setup vertices first."
				<< std::endl;

			return false;
		}
		osgCompute::Buffer::setDimension( 0, osg::Geometry::getVertexArray()->getNumElements() );


		osg::Geometry::ArrayList arrayList;
		osg::Geometry::getArrayList( arrayList );

		/////////////////
		// ELEMENTSIZE //
		/////////////////
		unsigned int elementSize = 0;
		for( unsigned int a=0; a<arrayList.size(); ++a )
		{
			// we assume that all arrays have the
			// same number of elements
			elementSize += (arrayList[a]->getTotalDataSize() / arrayList[a]->getNumElements());
		}
		setElementSize( elementSize );

		return osgCompute::Buffer::init();
	}

	//------------------------------------------------------------------------------
	void Geometry::releaseGLObjects( osg::State* state/*=0*/ ) const
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

		osg::Geometry::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void Geometry::drawImplementation( osg::RenderInfo& renderInfo ) const
	{
		const osgCompute::Context* curCtx = getContext( renderInfo.getState()->getContextID() );
		if( curCtx )
		{
			if( osgCompute::Buffer::getMapping( *curCtx ) != osgCompute::UNMAPPED )
				unmap( *curCtx );
		}

		osg::Geometry::drawImplementation( renderInfo );
	}

	//------------------------------------------------------------------------------
	void* Geometry::map( const osgCompute::Context& context, unsigned int mapping ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "Geometry::map() for geometry \""
				<< asObject()->getName() <<"\": geometry is dirty."
				<< std::endl;

			return NULL;
		}

		if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		{
			osg::notify(osg::FATAL)
				<< "Geometry::map() for geometry \""
				<< asObject()->getName() <<"\": calling thread differs from the context's thread."
				<< std::endl;

			return NULL;
		}

		GeometryStream* stream = static_cast<GeometryStream*>( osgCompute::Buffer::lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "Geometry::map() for geometry \""
				<< asObject()->getName() <<"\": could not receive geometry stream for context \""
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
	void Geometry::unmap( const osgCompute::Context& context ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Geometry::map() for geometry \""
				<< osg::Object::getName() <<"\": geometry is dirty."
				<< std::endl;

			return;
		}

		if( static_cast<const Context*>(&context)->getAssignedThread() != OpenThreads::Thread::CurrentThread() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Geometry::map() for geometry \""
				<< osg::Object::getName() <<"\": calling thread differs from the context's thread."
				<< std::endl;

			return;
		}

		GeometryStream* stream = static_cast<GeometryStream*>( osgCompute::Buffer::lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Geometry::map() for geometry \""
				<< osg::Object::getName() <<"\": could not receive geometry stream for context \""
				<< context.getId() << "\"."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::Geometry::setMemory( const osgCompute::Context& context, int value, unsigned int mapping, unsigned int offset, unsigned int count ) const
	{
		unsigned char* data = static_cast<unsigned char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Geometry::setMemory() for geometry \""
					<< osg::Object::getName() <<"\": error during memset() for host within context \""
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
					<< "osgCuda::Geometry::setMemory() for geometry \""
					<< osg::Object::getName() <<"\": error during cudaMemset() for device data within context \""
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
	void Geometry::clearLocal()
	{
	}

	//------------------------------------------------------------------------------
	void Geometry::clear( const osgCompute::Context& context ) const
	{
		if( osgCompute::Buffer::getMapping( context ) != osgCompute::UNMAPPED )
			unmap( context );

		osgCompute::Buffer::clear( context );
	}

	//------------------------------------------------------------------------------
	void* Geometry::mapStream( GeometryStream& stream, unsigned int mapping ) const
	{
		void* ptr = NULL;

		osgCuda::Geometry* thisGeom = const_cast<osgCuda::Geometry*>(this);
		osg::VertexBufferObject* vbo = thisGeom->getOrCreateVertexBufferObject();
		if( !vbo )
			return NULL;

		///////////////////
		// PROOF MAPPING //
		///////////////////
		if( ((stream._mapping & osgCompute::MAP_DEVICE && mapping & osgCompute::MAP_DEVICE) ||
			(stream._mapping & osgCompute::MAP_HOST && mapping & osgCompute::MAP_HOST)) )
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

			if( NULL == stream._devPtr )
			{
				cudaError res = cudaGLMapBufferObject( reinterpret_cast<void**>(&stream._devPtr), stream._bo );
				if( cudaSuccess != res )
				{
					osg::notify(osg::WARN)
						<< "osgCuda::Geometry::mapStream() for geometry \""<< osg::Object::getName()
						<< "\": error during cudaGLMapBufferObject() for context \""
						<< stream._context->getId()<<"\"."
						<< " " << cudaGetErrorString( res ) <<"."
						<< std::endl;

					return NULL;
				}
			}


			firstLoad = true;
		}

		/////////////
		// MAP VBO //
		/////////////
		if( NULL == stream._devPtr )
		{
			cudaError res = cudaGLMapBufferObject( reinterpret_cast<void**>(&stream._devPtr), stream._bo );
			if( cudaSuccess != res )
			{
				osg::notify(osg::WARN)
					<< "osgCuda::Geometry::mapStream() for geometry \""<< osg::Object::getName()
					<< "\": error during cudaGLMapBufferObject() for context \""
					<< stream._context->getId()<<"\"."
					<< " " << cudaGetErrorString( res )  <<"."
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
			if( vbo->isDirty( stream._context->getState()->getContextID() ) )
				setupStream( mapping, stream );

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
			if( vbo->isDirty( stream._context->getState()->getContextID() ) )
				setupStream( mapping, stream );

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
				<< "osgCuda::Geometry::mapStream() for geometry \""<< getName()<<"\": Wrong mapping type. Use one of the following types: "
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
	void Geometry::unmapStream( GeometryStream& stream ) const
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
					<< "osgCuda::Geometry::unmapStream() for geometry \""
					<< osg::Object::getName()<<"\": error during cudaGLUnmapBufferObject() for context \""
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
			stream._syncHost = true;
		}
		else if( (stream._mapping & osgCompute::MAP_HOST_TARGET) )
		{
			stream._syncDevice = true;
		}

		stream._mapping = osgCompute::UNMAPPED;
	}

	//------------------------------------------------------------------------------
	bool Geometry::setupStream( unsigned int mapping, GeometryStream& stream ) const
	{

		osgCuda::Geometry* thisGeom = const_cast<osgCuda::Geometry*>(this);
		osg::VertexBufferObject* vbo = thisGeom->getOrCreateVertexBufferObject();
		if( !vbo->isDirty( stream._context->getState()->getContextID() ) )
			return true;

		////////////////////
		// UNREGISTER VBO //
		////////////////////
		if( !stream._boRegistered )
			static_cast<const Context*>( stream._context.get() )->freeBufferObject( stream._bo );

		////////////////
		// UPDATE VBO //
		////////////////
		vbo->compileBuffer( const_cast<osg::State&>(*stream._context->getState()) );

		//////////////////
		// REGISTER VBO //
		//////////////////
		if( !static_cast<const Context*>( stream._context.get() )->registerBufferObject( stream._bo, osgCompute::Buffer::getByteSize() ) )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Geometry::setupStream() for geometry \""
				<< osg::Object::getName()<< "\": could not register buffer object for context \""
				<< stream._context->getId()<<"\"."
				<< std::endl;

			return false;
		}
		stream._boRegistered = true;
		stream._syncHost = true;

		return true;
	}

	//------------------------------------------------------------------------------
	bool Geometry::allocStream( unsigned int mapping, GeometryStream& stream ) const
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
						<< "osgCuda::Geometry::allocStream() for geometry \""
						<< osg::Object::getName()<<"\": something goes wrong within mallocDeviceHost() within context \""<<stream._context->getId()
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
						<< "osgCuda::Geometry::allocStream() for geometry \""
						<< osg::Object::getName()<<"\": something goes wrong within mallocHost() within context \""<<stream._context->getId()
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

			osgCuda::Geometry* thisGeom = const_cast<osgCuda::Geometry*>(this);
			osg::VertexBufferObject* vbo = thisGeom->getOrCreateVertexBufferObject();

			//////////////
			// SETUP BO //
			//////////////
			// compile buffer object if necessary
			if( vbo->isDirty(stream._context->getId()) )
				vbo->compileBuffer( const_cast<osg::State&>(*stream._context->getState()) );

			// using vertex buffers
			if( stream._bo == UINT_MAX )
				stream._bo = vbo->buffer(stream._context->getId());

			//////////////////
			// REGISTER PBO //
			//////////////////
			if( !stream._boRegistered )
			{
				if( !static_cast<const Context*>( stream._context.get() )->registerBufferObject( stream._bo, osgCompute::Buffer::getByteSize() ) )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::Geometry::allocStream() for geometry \""
						<< osg::Object::getName()<< "\": could not register buffer object for context \""
						<< stream._context->getId()<<"\"."
						<< std::endl;

					return false;
				}
			}
			stream._boRegistered = true;

			if( stream._hostPtr != NULL )
				stream._syncDevice = true;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	bool Geometry::syncStream( unsigned int mapping, GeometryStream& stream ) const
	{
		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			res = cudaMemcpy( stream._devPtr, stream._hostPtr, osgCompute::Buffer::getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::Geometry::syncStream() for geometry \""<< asObject()->getName()
					<< "\": error during cudaMemcpy() to device within context \""
					<< stream._context->getId() << "\". "
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
					<< "osgCuda::Geometry::syncStream() for geometry \""
					<< asObject()->getName()<<"\": something goes wrong within cudaMemcpy() to host within context \""
					<< stream._context->getId() << "\". "
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
	osgCompute::BufferStream* Geometry::newStream( const osgCompute::Context& context ) const
	{
		return new GeometryStream;
	}
}
