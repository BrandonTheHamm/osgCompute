#include <memory.h>
#include <malloc.h>
#include <osg/GL>
#include <osg/RenderInfo>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Context>
#include <osgCuda/Geometry>

namespace osgCuda
{
    /**
    */
    class LIBRARY_EXPORT GeometryStream : public osgCompute::BufferStream
    {
    public:
        void*					 _hostPtr;
        bool                     _hostPtrAllocated;
        bool                     _syncHost;
        void*					 _devPtr;
        bool                     _syncDevice;
        GLuint                   _bo;
        bool                     _boRegistered;

        GeometryStream();
        virtual ~GeometryStream();


    private:
        // not allowed to call copy-constructor or copy-operator
        GeometryStream( const GeometryStream& ) {}
        GeometryStream& operator=( const GeometryStream& ) { return *this; }
    };

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
		{
			cudaError_t res = cudaGLUnregisterBufferObject( _bo );
			if( res != cudaSuccess )
				osg::notify(osg::FATAL)
				<<"TextureStream::~TextureStream(): error during cudaGLUnregisterBufferObject()."
				<< cudaGetErrorString(res) << std::endl;
		}

		if( _hostPtrAllocated && NULL != _hostPtr)
			free( _hostPtr );
	}

	/**
    */
    class LIBRARY_EXPORT GeometryBuffer : public osgCompute::InteropBuffer
    {
    public:
        GeometryBuffer();

		META_Object(osgCuda,GeometryBuffer)

		virtual osgCompute::InteropObject* getObject() { return _geomref.get(); }

        virtual bool init();

        virtual bool setMemory( const osgCompute::Context& context, int value, unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int offset = 0, unsigned int count = UINT_MAX, unsigned int hint = 0 ) const;
        virtual void* map( const osgCompute::Context& context, unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int offset = 0, unsigned int hint = 0 ) const;
        virtual void unmap( const osgCompute::Context& context, unsigned int hint = 0 ) const;

		virtual void clear( const osgCompute::Context& context ) const;
        virtual void clear();
    protected:
		friend class Geometry;
        virtual ~GeometryBuffer();
        void clearLocal();

        virtual void* mapStream( GeometryStream& stream, unsigned int mapping, unsigned int offset ) const;
        virtual void unmapStream( GeometryStream& stream ) const;

        bool setupStream( unsigned int mapping, GeometryStream& stream ) const;
        bool allocStream( unsigned int mapping, GeometryStream& stream ) const;
        bool syncStream( unsigned int mapping, GeometryStream& stream ) const;

        virtual osgCompute::BufferStream* newStream( const osgCompute::Context& context ) const;

		osg::ref_ptr<osgCuda::Geometry> _geomref;
    private:
        // copy constructor and operator should not be called
        GeometryBuffer( const GeometryBuffer& , const osg::CopyOp& ) {}
        GeometryBuffer& operator=(const GeometryBuffer&) { return (*this); }
    };

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	GeometryBuffer::GeometryBuffer()
		:  osgCompute::InteropBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	GeometryBuffer::~GeometryBuffer()
	{
		clearLocal();

		// proxy is now deleted
		_geomref->_proxy = NULL;
		// attach handles
		_geomref->_handles = getHandles();
		// decrease reference count of geometry reference
		_geomref = NULL;
	}

	//------------------------------------------------------------------------------
	void GeometryBuffer::clear()
	{
		osgCompute::Buffer::clear();
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void GeometryBuffer::clear( const osgCompute::Context& context ) const
	{
		if( getMapping( context ) != osgCompute::UNMAPPED )
			unmap( context );

		osgCompute::Buffer::clear( context );
	}

	//------------------------------------------------------------------------------
	bool GeometryBuffer::init()
	{
		if( !osgCompute::Resource::isClear() )
			return true;

		if( !_geomref.valid() )
			return false;

		if( _geomref->getVertexArray() == NULL || _geomref->getVertexArray()->getNumElements() == 0 )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::GeometryBuffer::initDimension(): no dimensions defined for geometry! setup vertex array first."
				<< std::endl;

			return false;
		}
		setDimension( 0, _geomref->getVertexArray()->getNumElements() );


		osg::Geometry::ArrayList arrayList;
		_geomref->getArrayList( arrayList );

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
	void* GeometryBuffer::map( const osgCompute::Context& context, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "GeometryBuffer::map(): geometry is dirty."
				<< std::endl;

			return NULL;
		}

		if( !context.isConnectedWithGraphicsContext() )
		{
			osg::notify(osg::FATAL)
				<< "GeometryBuffer::map(): context is not connected with a graphics context."
				<< std::endl;

			return NULL;
		}

		GeometryStream* stream = static_cast<GeometryStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "GeometryBuffer::map(): cannot receive geometry stream."
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
	void GeometryBuffer::unmap( const osgCompute::Context& context, unsigned int ) const
	{
		if( osgCompute::Resource::isClear() )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::GeometryBuffer::map(): geometry buffer is dirty."
				<< std::endl;

			return;
		}

		GeometryStream* stream = static_cast<GeometryStream*>( lookupStream(context) );
		if( NULL == stream )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::GeometryBuffer::map(): cannot receive geometry stream."
				<< std::endl;

			return;
		}

		unmapStream( *stream );
	}

	//------------------------------------------------------------------------------
	bool osgCuda::GeometryBuffer::setMemory( const osgCompute::Context& context, int value, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int count/* = UINT_MAX*/, unsigned int ) const
	{
		unsigned char* data = static_cast<unsigned char*>( map( context, mapping ) );
		if( NULL == data )
			return false;

		if( mapping & osgCompute::MAP_HOST_TARGET )
		{
			if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::GeometryBuffer::setMemory(): error during memset() for host."
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
					<< "osgCuda::GeometryBuffer::setMemory(): error during cudaMemset() for device data."
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
	void GeometryBuffer::clearLocal()
	{
	}

	//------------------------------------------------------------------------------
	void* GeometryBuffer::mapStream( GeometryStream& stream, unsigned int mapping, unsigned int offset ) const
	{
		void* ptr = NULL;

		osg::VertexBufferObject* vbo =  _geomref.get()->getOrCreateVertexBufferObject();
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

			if( NULL == stream._devPtr )
			{
				cudaError res = cudaGLMapBufferObject( reinterpret_cast<void**>(&stream._devPtr), stream._bo );
				if( cudaSuccess != res )
				{
					osg::notify(osg::WARN)
						<< "osgCuda::GeometryBuffer::mapStream() : error during cudaGLMapBufferObject(). "
						<< cudaGetErrorString( res ) <<"."
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
					<< "osgCuda::GeometryBuffer::mapStream(): error during cudaGLMapBufferObject(). "
					<< cudaGetErrorString( res )  <<"."
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
			if( vbo->isDirty( stream._context->getGraphicsContext()->getState()->getContextID() ) )
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
			if( vbo->isDirty( stream._context->getGraphicsContext()->getState()->getContextID() ) )
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
				<< "osgCuda::GeometryBuffer::mapStream(): Wrong mapping type specified. Use one of the following types: "
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
	void GeometryBuffer::unmapStream( GeometryStream& stream ) const
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
					<< "osgCuda::GeometryBuffer::unmapStream(): error during cudaGLUnmapBufferObject(). "
					<< cudaGetErrorString( res ) <<"."
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
	bool GeometryBuffer::setupStream( unsigned int mapping, GeometryStream& stream ) const
	{
		osg::State* state = stream._context->getGraphicsContext()->getState();
		if( state == NULL )
			return false;

		osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
		if( !vbo->isDirty( state->getContextID() ) )
			return true;

		////////////////////
		// UNREGISTER VBO //
		////////////////////
		if( !stream._boRegistered )
		{
			cudaError_t res = cudaGLUnregisterBufferObject ( stream._bo );
			if( res != cudaSuccess )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::GeometryBuffer::setupStream(): unable to unregister buffer object. "
					<< std::endl;

				return false;
			}
		}
		stream._boRegistered = false;

		////////////////
		// UPDATE VBO //
		////////////////
		vbo->compileBuffer( const_cast<osg::State&>(*state) );

		//////////////////
		// REGISTER VBO //
		//////////////////
		cudaError_t res = cudaGLRegisterBufferObject( stream._bo );
		if( res != cudaSuccess )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::GeometryBuffer::setupStream(): unable to register buffer object again."
				<< std::endl;

			return false;
		}
		stream._boRegistered = true;
		stream._syncHost = true;

		return true;
	}

	//------------------------------------------------------------------------------
	bool GeometryBuffer::allocStream( unsigned int mapping, GeometryStream& stream ) const
	{
		if( mapping & osgCompute::MAP_HOST )
		{
			if( stream._hostPtr != NULL )
				return true;

			stream._hostPtr = malloc( getByteSize() );
			if( NULL == stream._hostPtr )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::GeometryBuffer::allocStream(): error during mallocHost()."
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

			osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();

			osg::State* state = stream._context->getGraphicsContext()->getState();
			if( state == NULL )
				return false;

			//////////////
			// SETUP BO //
			//////////////
			// compile buffer object if necessary
			if( vbo->isDirty(state->getContextID()) )
				vbo->compileBuffer( const_cast<osg::State&>(*state) );

			// using vertex buffers
			if( stream._bo == UINT_MAX )
				stream._bo = vbo->buffer(state->getContextID());

			//////////////////
			// REGISTER PBO //
			//////////////////
			if( !stream._boRegistered )
			{
				cudaError_t res = cudaGLRegisterBufferObject( stream._bo );
				if( res != cudaSuccess )
				{
					osg::notify(osg::FATAL)
						<< "osgCuda::GeometryBuffer::allocStream(): unable to register buffer object."
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
	bool GeometryBuffer::syncStream( unsigned int mapping, GeometryStream& stream ) const
	{
		cudaError res;
		if( mapping & osgCompute::MAP_DEVICE )
		{
			res = cudaMemcpy( stream._devPtr, stream._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
			if( cudaSuccess != res )
			{
				osg::notify(osg::FATAL)
					<< "osgCuda::GeometryBuffer::syncStream(): error during cudaMemcpy() to device. "
					<< cudaGetErrorString( res ) <<"."
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
					<< "osgCuda::GeometryBuffer::syncStream(): error during cudaMemcpy() to host. "
					<< cudaGetErrorString( res ) <<"."
					<< std::endl;

				return false;
			}

			stream._syncHost = false;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------
	osgCompute::BufferStream* GeometryBuffer::newStream( const osgCompute::Context& context ) const
	{
		return new GeometryStream;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Geometry::Geometry()
		: osg::Geometry(),
		  _proxy(NULL)
	{
		// geometry must use vertex buffer objects
		setUseVertexBufferObjects( true );
	}

	//------------------------------------------------------------------------------
	bool Geometry::init()
	{
		if( NULL != _proxy )
			_proxy->init();

		_clear = false;
		return true;
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Geometry::getBuffer()
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	const osgCompute::InteropBuffer* Geometry::getBuffer() const
	{
		return _proxy;
	}

	//------------------------------------------------------------------------------
	osgCompute::InteropBuffer* Geometry::getOrCreateBuffer()
	{
		// create proxy buffer on demand
		if( _proxy == NULL )
		{
			_proxy = new GeometryBuffer;
			_proxy->_geomref = this;
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
	void Geometry::addHandle( const std::string& handle )
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
	void Geometry::removeHandle( const std::string& handle )
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
	bool Geometry::isAddressedByHandle( const std::string& handle ) const
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
	void Geometry::releaseGLObjects( osg::State* state/*=0*/ ) const
	{
		if( state != NULL )
		{
			const osgCompute::Context* curCtx = osgCompute::Context::getContext( state->getContextID() );
			if( NULL != curCtx && NULL != _proxy )
			{
				if( _proxy->getMapping( *curCtx ) != osgCompute::UNMAPPED )
					_proxy->unmap( *curCtx );

				_proxy->clear( *curCtx );
			}
		}

		osg::Geometry::releaseGLObjects( state );
	}

	//------------------------------------------------------------------------------
	void Geometry::drawImplementation( osg::RenderInfo& renderInfo ) const
	{
		const osgCompute::Context* curCtx = osgCompute::Context::getContext( renderInfo.getState()->getContextID() );
		if( NULL != curCtx && NULL != _proxy )
		{
			if( _proxy->getMapping( *curCtx ) != osgCompute::UNMAPPED )
				_proxy->unmap( *curCtx );
		}

		osg::Geometry::drawImplementation( renderInfo );
	}

	//------------------------------------------------------------------------------
	bool Geometry::isClear()
	{
		return _clear;
	}

	//------------------------------------------------------------------------------
	void Geometry::clear()
	{
		clearLocal();
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	Geometry::~Geometry()
	{
		if( _proxy != NULL )
		{
			osg::notify(osg::FATAL)
				<< "osgCuda::Geometry::destructor(): proxy is still valid!!!."
				<< std::endl;
		}
	}

	//------------------------------------------------------------------------------
	void Geometry::clearLocal()
	{
		_clear = true;
		if( NULL != _proxy )
			_proxy->clear();

		_handles.clear();
	}
}
