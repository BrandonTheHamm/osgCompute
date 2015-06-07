#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
#include <driver_types.h>
#include <osg/GL>
#include <osg/RenderInfo>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osg/observer_ptr>
#include <osgCompute/Memory>
#include <osgCuda/Geometry>

namespace osgCuda
{
    /**
    */
    class LIBRARY_EXPORT GeometryObject : public osgCompute::MemoryObject
    {
    public:
        void*						_hostPtr;
        void*						_devPtr;
        cudaGraphicsResource*       _graphicsResource;
        std::vector<unsigned int>	_lastModifiedCount;

        GeometryObject();
        virtual ~GeometryObject();


    private:
        // not allowed to call copy-constructor or copy-operator
        GeometryObject( const GeometryObject& ) {}
        GeometryObject& operator=( const GeometryObject& ) { return *this; }
    };


    /**
    */
    class LIBRARY_EXPORT IndexedGeometryObject : public GeometryObject
    {
    public:
        void*						_hostIdxPtr;
        void*						_devIdxPtr;
        cudaGraphicsResource*       _graphicsIdxResource;
        std::vector<unsigned int>	_lastIdxModifiedCount;
        unsigned int                _syncIdxOp;
        unsigned int                _idxMapping;


        IndexedGeometryObject();
        virtual ~IndexedGeometryObject();


    private:
        // not allowed to call copy-constructor or copy-operator
        IndexedGeometryObject( const IndexedGeometryObject& ) {}
        IndexedGeometryObject& operator=( const IndexedGeometryObject& ) { return *this; }
    };


    /**
    */
    class LIBRARY_EXPORT GeometryMemory : public osgCompute::GLMemory
    {
    public:
        GeometryMemory();

        META_Object(osgCuda,GeometryMemory)

		virtual osgCompute::GLMemoryAdapter* getAdapter(); 
		virtual const osgCompute::GLMemoryAdapter* getAdapter() const; 

        virtual void* map( unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int offset = 0, unsigned int hint = 0 );
        virtual void unmap( unsigned int hint = 0 );
        virtual bool reset( unsigned int hint = 0 );
        virtual bool supportsMapping( unsigned int mapping, unsigned int hint = 0 ) const;
        virtual void mapAsRenderTarget();
        virtual unsigned int getAllocatedByteSize( unsigned int mapping, unsigned int hint = 0 ) const;
        virtual unsigned int getByteSize( unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int hint = 0 ) const;

        virtual unsigned int getElementSize() const;
        virtual unsigned int getDimension( unsigned int dimIdx ) const;
        virtual unsigned int getNumDimensions() const;
        virtual unsigned int getNumElements() const;

    protected:
        friend class Geometry;
        virtual ~GeometryMemory();


        bool setup( unsigned int mapping );
        bool alloc( unsigned int mapping );
        bool sync( unsigned int mapping );

        virtual osgCompute::MemoryObject* createObject() const;
        virtual unsigned int computePitch() const;

        osg::observer_ptr<osgCuda::Geometry>		_geomref;
    private:
        // copy constructor and operator should not be called
        GeometryMemory( const GeometryMemory& , const osg::CopyOp& ) {}
        GeometryMemory& operator=(const GeometryMemory&) { return (*this); }
    };

    /**
    */
    class LIBRARY_EXPORT IndexedGeometryMemory : public GeometryMemory
    {
    public:
        IndexedGeometryMemory();

        META_Object(osgCuda,IndexedGeometryMemory)

        virtual void* map( unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int offset = 0, unsigned int hint = 0 );
        virtual void unmap( unsigned int hint = 0 );
        virtual bool reset( unsigned int hint = 0 );
        virtual bool supportsMapping( unsigned int mapping, unsigned int hint = 0 ) const;

        virtual void* mapIndices( unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int offset = 0, unsigned int hint = 0 );
        virtual void unmapIndices( unsigned int hint = 0 );
        virtual bool resetIndices( unsigned int hint = 0 );

        virtual unsigned int getIndicesByteSize() const;
        virtual void clear();

    protected:
        friend class Geometry;
        virtual ~IndexedGeometryMemory();


        bool setupIndices( unsigned int mapping );
        bool allocIndices( unsigned int mapping );
        bool syncIndices( unsigned int mapping );

        mutable unsigned int                        _indicesByteSize;

        virtual osgCompute::MemoryObject* createObject() const;
    private:
        // copy constructor and operator should not be called
        IndexedGeometryMemory( const IndexedGeometryMemory& , const osg::CopyOp& ) {}
        IndexedGeometryMemory& operator=(const IndexedGeometryMemory&) { return (*this); }
    };


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    GeometryObject::GeometryObject()
	:	osgCompute::MemoryObject(),
		_hostPtr(NULL),
        _devPtr(NULL),
        _graphicsResource( NULL )
    {
        _lastModifiedCount.clear();
    }

    //------------------------------------------------------------------------------
    GeometryObject::~GeometryObject()
    {
        if( _devPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &_graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
        }

        if( _graphicsResource != NULL )
        {
            cudaError_t res = cudaGraphicsUnregisterResource( _graphicsResource );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                <<__FUNCTION__ << ": error during cudaGraphicsUnregisterResource()."
                << cudaGetErrorString(res) << std::endl;
            }
        }

        if( NULL != _hostPtr)
            free( _hostPtr );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    IndexedGeometryObject::IndexedGeometryObject()
	:	GeometryObject(),
			_hostIdxPtr(NULL),
            _devIdxPtr(NULL),
			_graphicsIdxResource( NULL ),
			_syncIdxOp( osgCompute::NO_SYNC ),
            _idxMapping( osgCompute::UNMAP )
    {   
        _lastIdxModifiedCount.clear();
    }

    //------------------------------------------------------------------------------
    IndexedGeometryObject::~IndexedGeometryObject()
    {
        if( _devIdxPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &_graphicsIdxResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    <<__FUNCTION__ << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
        }

        if( _graphicsIdxResource != NULL )
        {
            cudaError_t res = cudaGraphicsUnregisterResource( _graphicsIdxResource );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    <<__FUNCTION__ << ": error during cudaGraphicsUnregisterResource()."
                    << cudaGetErrorString(res) << std::endl;
            }
        }

        if( NULL != _hostIdxPtr)
            free( _hostIdxPtr );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    GeometryMemory::GeometryMemory()
        :  osgCompute::GLMemory()
    {
        // Please note that virtual functions className() and libraryName() are called
        // during observeResource() which will only develop until this class.
        // However if contructor of a subclass calls this function again observeResource
        // will change the className and libraryName of the observed pointer.
        osgCompute::ResourceObserver::instance()->observeResource( *this );
    }

    //------------------------------------------------------------------------------
    GeometryMemory::~GeometryMemory()
    {
    }

    //------------------------------------------------------------------------------
    osgCompute::GLMemoryAdapter* GeometryMemory::getAdapter()
    { 
        return _geomref.get(); 
    }

	//------------------------------------------------------------------------------
	const osgCompute::GLMemoryAdapter* GeometryMemory::getAdapter() const
	{ 
		return _geomref.get(); 
	}

    //------------------------------------------------------------------------------
    void GeometryMemory::mapAsRenderTarget()
    {
        // Do nothing as geometry cannot be mapped as a render target.
    }
    
    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::getElementSize() const 
    { 
        unsigned int elementSize = osgCompute::Memory::getElementSize();
        if( elementSize == 0 )
        {
            if( !_geomref.valid() )
                return 0;

            osg::Geometry::ArrayList arrayList;
            _geomref->getArrayList( arrayList );

            elementSize = 0;
            for( unsigned int a=0; a<arrayList.size(); ++a )
            {
                if( arrayList[a] != NULL )
                {
                    // we assume that all arrays have the
                    // same number of elements
                    elementSize += (arrayList[a]->getTotalDataSize() / arrayList[a]->getNumElements());
                }
            }

            const_cast<osgCuda::GeometryMemory*>(this)->setElementSize( elementSize );
        }

        return elementSize; 
    }

    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::getNumDimensions() const
    {        
        unsigned int numDims = osgCompute::Memory::getNumDimensions();
        if( numDims == 0 )
        {
            if( !_geomref.valid() )
                return 0;

            if( _geomref->getVertexArray() == NULL || _geomref->getVertexArray()->getNumElements() == 0 )
                return 0;

            const_cast<osgCuda::GeometryMemory*>(this)->setDimension( 0, _geomref->getVertexArray()->getNumElements() );
            numDims = osgCompute::Memory::getNumDimensions();
        }

        return numDims;
    }

    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::getDimension( unsigned int dimIdx ) const
    { 
        if( osgCompute::Memory::getNumDimensions() == 0 )
        {
            if( !_geomref.valid() )
                return 0;

            if( _geomref->getVertexArray() == NULL || _geomref->getVertexArray()->getNumElements() == 0 )
                return 0;

            const_cast<osgCuda::GeometryMemory*>(this)->setDimension( 0, _geomref->getVertexArray()->getNumElements() );
        }

        return osgCompute::Memory::getDimension(dimIdx);
    }

    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::getNumElements() const
    {
        unsigned int numElements = osgCompute::Memory::getNumElements();
        if( numElements == 0 )
        {
            if( !_geomref.valid() )
                return 0;

            if( _geomref->getVertexArray() == NULL || _geomref->getVertexArray()->getNumElements() == 0 )
                return 0;

            const_cast<osgCuda::GeometryMemory*>(this)->setDimension( 0, _geomref->getVertexArray()->getNumElements() );
            numElements = osgCompute::Memory::getNumElements();
        }

        return numElements;
    }

    //------------------------------------------------------------------------------
    void* GeometryMemory::map( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint/* = 0*/ )
    {
        if( !_geomref.valid() )
			return NULL;

        if( mapping == osgCompute::UNMAP )
        {
            unmap( hint );
            return NULL;
        }

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object(true) );
        if( !memoryPtr )
            return NULL;
        GeometryObject& memory = *memoryPtr;

        //////////////
        // MAP DATA //
        //////////////
        void* ptr = NULL;

        osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
        if( !vbo )
            return NULL;


        memory._mapping = mapping;
        bool firstLoad = false;
        bool needsSetup = false;


        if( mapping & osgCompute::MAP_HOST )
        {
            //////////////////////////
            // ALLOCATE HOST-MEMORY //
            //////////////////////////
            if( NULL == memory._hostPtr )
            {
                if( !alloc( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            // check if buffers have changed
            if( memory._lastModifiedCount.size() != vbo->getNumBufferData() )
                needsSetup = true;
            else
                for( unsigned int d=0; d<memory._lastModifiedCount.size(); ++d )
                    if( memory._lastModifiedCount[d] != vbo->getBufferData(d)->getModifiedCount() )
                        needsSetup = true;

            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup && !(memory._syncOp & osgCompute::SYNC_HOST) )
            {
                if( !setup( mapping ) )
                    return NULL;
            }

            /////////////////
            // SYNC STREAM //
            /////////////////
            if( memory._syncOp & osgCompute::SYNC_HOST )
            {
                // map's device ptr if necessary
                if( !sync( mapping ) )
                    return NULL;
            }

            ptr = memory._hostPtr;
        }
        else if( (mapping & osgCompute::MAP_DEVICE) )
        {
            if( osgCompute::GLMemory::getContext() == NULL || osgCompute::GLMemory::getContext()->getState() == NULL )
                return NULL;

            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !glBO )
                return NULL;

            needsSetup = glBO->isDirty();

            ////////////////////////////
            // ALLOCATE DEVICE-MEMORY //
            ////////////////////////////
            // create dynamic texture device memory
            // for each type of mapping
            if( NULL == memory._graphicsResource )
            {
                if( !alloc( osgCompute::MAP_DEVICE ) )
                    return NULL;

                firstLoad = true;
            }

            /////////////
            // MAP VBO //
            /////////////
            if( NULL == memory._devPtr )
            {
                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }

                size_t memSize = 0;
                res = cudaGraphicsResourceGetMappedPointer(&memory._devPtr, &memSize, memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }
            }

            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup )
                if( !setup( mapping ) )
                    return NULL;

            /////////////////
            // SYNC STREAM //
            /////////////////
            if( ( memory._syncOp & osgCompute::SYNC_DEVICE ) && NULL != memory._hostPtr )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._devPtr;
        }
        else
        {
            osg::notify(osg::WARN)
                << __FUNCTION__ <<" " << _geomref->getName() << ": wrong mapping type specified. Use one of the following types: "
                << "HOST_SOURCE, HOST_TARGET, HOST, DEVICE_SOURCE, DEVICE_TARGET, DEVICE."
                << std::endl;

            return NULL;
        }

        if( NULL ==  ptr )
            return NULL;

        //////////////////
        // LOAD/SUBLOAD //
        //////////////////
        if( getSubloadCallback() && NULL != ptr )
        {
            const osgCompute::SubloadCallback* callback = getSubloadCallback();
            if( callback )
            {
                // load or subload data before returning the pointer
                if( firstLoad )
                    callback->load( ptr, mapping, offset, *this );
                else
                    callback->subload( ptr, mapping, offset, *this );
            }
        }

        if( (mapping & osgCompute::MAP_DEVICE_TARGET) == osgCompute::MAP_DEVICE_TARGET )
            memory._syncOp |= osgCompute::SYNC_HOST;

        if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
            memory._syncOp |= osgCompute::SYNC_DEVICE;

        return &static_cast<char*>(ptr)[offset];
    }

    //------------------------------------------------------------------------------
    void GeometryMemory::unmap( unsigned int )
    {
		if( !_geomref.valid() )
			return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object(false) );
        if( !memoryPtr )
            return;
        GeometryObject& memory = *memoryPtr;


        if( memory._graphicsResource == NULL )
        {
            // Geometry object will be created during rendering
            // so update the host memory during next mapping
            memory._syncOp |= osgCompute::SYNC_HOST;
        }

        //////////////////
        // UNMAP MEMORY //
        //////////////////
        // Copy host memory to VBO
        if( memory._syncOp & osgCompute::SYNC_DEVICE )
        {
            // Will remove sync flag
            if( NULL == map( osgCompute::MAP_DEVICE_SOURCE, 0 ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during device memory synchronization (map())."
                    << std::endl;

                return;
            }
        }

        // Change current context to render context
        if( memory._devPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            memory._devPtr = NULL;
            memory._mapping = osgCompute::UNMAP;
        }
    }

    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::getAllocatedByteSize( unsigned int mapping, unsigned int hint /*= 0 */ ) const 
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        const GeometryObject* memoryPtr = dynamic_cast<const GeometryObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        const GeometryObject& memory = *memoryPtr;

        unsigned int allocSize = 0;
        switch( mapping )
        {
        case osgCompute::MAP_DEVICE: case osgCompute::MAP_DEVICE_TARGET: case osgCompute::MAP_DEVICE_SOURCE:
            {
                allocSize = (memory._graphicsResource != NULL)? getByteSize( mapping, hint ) : 0;
            }break;
        case osgCompute::MAP_HOST: case osgCompute::MAP_HOST_TARGET: case osgCompute::MAP_HOST_SOURCE:
            {
                allocSize = (memory._hostPtr != NULL)? getByteSize( mapping, hint ) : 0;
            }break;
        case osgCompute::MAP_DEVICE_ARRAY: case osgCompute::MAP_DEVICE_ARRAY_TARGET:
            {
                allocSize = 0;
            }break;
        }

        return allocSize;
    }

    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::getByteSize( unsigned int mapping, unsigned int hint /*= 0 */ ) const
    {
        unsigned int allocSize = 0;
        switch( mapping )
        {
        case osgCompute::MAP_DEVICE: case osgCompute::MAP_DEVICE_TARGET: case osgCompute::MAP_DEVICE_SOURCE:
            {
                allocSize =  getElementSize() * getNumElements();

            }break;
        case osgCompute::MAP_HOST: case osgCompute::MAP_HOST_TARGET: case osgCompute::MAP_HOST_SOURCE:
            {
                allocSize = getElementSize() * getNumElements();

            }break;
        case osgCompute::MAP_DEVICE_ARRAY: case osgCompute::MAP_DEVICE_ARRAY_TARGET:
            {
                allocSize = 0;
            }break;
        }

        return allocSize;
    }

    //------------------------------------------------------------------------------
    bool GeometryMemory::reset( unsigned int  )
	{		
		if( !_geomref.valid() )
			return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object(false) );
        if( !memoryPtr )
            return false;
        GeometryObject& memory = *memoryPtr;

        ////////////////////////
        // CLEAR MEMORY FIRST //
        ////////////////////////
        if( memory._hostPtr != NULL )
        {
            if( !memset( memory._hostPtr, 0x0, getAllElementsSize() ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during memset() for host memory."
                    << std::endl;

                return false;
            }
        }

        // clear device memory
        if( memory._graphicsResource != NULL )
        {
            if( memory._devPtr == NULL )
            {
                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }

                size_t memSize = 0;
                res = cudaGraphicsResourceGetMappedPointer(&memory._devPtr, &memSize, memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }
            }

            cudaError res = cudaMemset( memory._devPtr, 0x0, getAllElementsSize() );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaMemset(). "
                    << cudaGetErrorString( res )  <<"."
                    << std::endl;

                return false;
            }
        }

        ///////////////////////
        // UNREGISTER BUFFER //
        ///////////////////////
        if( memory._devPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    <<  __FUNCTION__ <<": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return false;
            }

            memory._devPtr = NULL;
        }

        if( memory._graphicsResource != NULL )
        {
            cudaError_t res = cudaGraphicsUnregisterResource( memory._graphicsResource );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<": error during cudaGraphicsUnregisterResource()."
                    << cudaGetErrorString(res) << std::endl;
                return false;
            }

            memory._graphicsResource = NULL;
        }


        //////////////////
        // RESET MEMORY //
        //////////////////
        // Reset array data
        memory._lastModifiedCount.clear();
        memory._syncOp = osgCompute::NO_SYNC;

        osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
        if( !vbo )
        {
            osg::notify(osg::FATAL)
                << __FUNCTION__ <<" " << _geomref->getName() << ": no buffer object found."
                << std::endl;

            return false;
        }

        vbo->dirty();

        // Compile vertex buffer
        osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
        if( NULL == glBO )
        {
            osg::notify(osg::FATAL)
                << __FUNCTION__ <<" " << _geomref->getName() << ": no GL buffer object found."
                << std::endl;

            return false;
        }

        glBO->dirty();

        return true;
    }



    //------------------------------------------------------------------------------
    bool GeometryMemory::supportsMapping( unsigned int mapping, unsigned int ) const
    {
        switch( mapping )
        {
        case osgCompute::UNMAP:
        case osgCompute::MAP_HOST:
        case osgCompute::MAP_HOST_SOURCE:
        case osgCompute::MAP_HOST_TARGET:
        case osgCompute::MAP_DEVICE:
        case osgCompute::MAP_DEVICE_SOURCE:
        case osgCompute::MAP_DEVICE_TARGET:
            return true;
        default:
            return false;
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    unsigned int GeometryMemory::computePitch() const
    {
        return getDimension(0)*getElementSize();
    }

    //------------------------------------------------------------------------------
    bool GeometryMemory::setup( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        GeometryObject& memory = *memoryPtr;


        //////////////////
        // SETUP MEMORY //
        //////////////////
        osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
        if( mapping & osgCompute::MAP_DEVICE )
        {
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !glBO->isDirty() )
                return true;

            ////////////////////
            // UNREGISTER VBO //
            ////////////////////
            bool wasRegistered = false;
            if( memory._graphicsResource != NULL )
            {
                wasRegistered = true;
                cudaError res = cudaGraphicsUnregisterResource( memory._graphicsResource );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": unable to unregister buffer object. "
                        << std::endl;

                    return false;
                }
                memory._graphicsResource = NULL;
            }

            ////////////////
            // UPDATE VBO //
            ////////////////
            if( glBO->isDirty() )
                glBO->compileBuffer();

            //////////////////
            // REGISTER VBO //
            //////////////////
            if( wasRegistered )
            {
                cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": unable to register buffer object again."
                        << std::endl;

                    return false;
                }
            }

            if( (memory._syncOp & osgCompute::SYNC_DEVICE) == osgCompute::SYNC_DEVICE )
                memory._syncOp ^= osgCompute::SYNC_DEVICE;

            memory._syncOp = osgCompute::SYNC_HOST;
        }
        else //  mapping & osgCompute::MAP_HOST
        {
            unsigned char* hostPtr = static_cast<unsigned char*>( memory._hostPtr );

            // Copy buffers into host memory
            if( memory._lastModifiedCount.size() != vbo->getNumBufferData() )
                memory._lastModifiedCount.resize( vbo->getNumBufferData(), UINT_MAX );

            unsigned int curOffset = 0;
            for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
            {
                osg::BufferData* curData = vbo->getBufferData(d);
                if( !curData )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": invalid buffer data found."
                        << std::endl;

                    return false;
                }

                if( curData->getModifiedCount() != memory._lastModifiedCount[d] )
                {
                    // Copy memory
                    memcpy( &hostPtr[curOffset], curData->getDataPointer(), curData->getTotalDataSize() );

                    // Store last modified value
                    memory._lastModifiedCount[d] = curData->getModifiedCount();
                }
                curOffset += curData->getTotalDataSize();
            }

            if( (memory._syncOp & osgCompute::SYNC_HOST) == osgCompute::SYNC_HOST )
                memory._syncOp ^= osgCompute::SYNC_HOST;

            memory._syncOp = osgCompute::SYNC_DEVICE;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool GeometryMemory::alloc( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object(true) );
        if( !memoryPtr )
            return NULL;
        GeometryObject& memory = *memoryPtr;

        //////////////////////////////
        // ALLOCATE/REGSITER MEMORY //
        //////////////////////////////
        if( mapping & osgCompute::MAP_HOST )
        {
            if( memory._hostPtr != NULL )
                return true;

            memory._hostPtr = malloc( getAllElementsSize() );
            if( NULL == memory._hostPtr )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during mallocHost()."
                    << std::endl;

                return false;
            }

            if( memory._devPtr != NULL || 
                (memory._syncOp & osgCompute::SYNC_HOST) )
            {
                memory._syncOp |= osgCompute::SYNC_HOST;

                // synchronize host memory with device memory and avoid copying data
                // from buffers in first place
                memory._lastModifiedCount.clear();
                osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
                for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
                    memory._lastModifiedCount.push_back( vbo->getBufferData(d)->getModifiedCount() );
            }
            else
            {
                // Mark buffers to be copied into the host memory
                memory._lastModifiedCount.clear();
                osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
                for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
                    memory._lastModifiedCount.push_back( UINT_MAX );

                memory._syncOp |= osgCompute::SYNC_DEVICE;
            }
            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
            if( !vbo )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": no buffer object found."
                    << std::endl;

                return false;
            }

            // Compile vertex buffer
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !glBO )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": no GL buffer object found."
                    << std::endl;

                return false;
            }

            if( glBO->isDirty() )
            {
                osg::GLExtensions* ext = osg::GLExtensions::Get(osgCompute::GLMemory::getContext()->getState()->getContextID(), true);
                if( !ext )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": cannot find required extensions to compile buffer object."
                        << std::endl;

                    return false;
                }

                glBO->compileBuffer();
                // Unbind buffer objects
                ext->glBindBuffer(GL_ARRAY_BUFFER_ARB,0);
                ext->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB,0);
            }

            // Avoid copy operation during osgCompute::MAP_HOST
            memory._lastModifiedCount.clear();
            for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
                memory._lastModifiedCount.push_back( vbo->getBufferData(d)->getModifiedCount() );

            // Register vertex buffer 
            cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": unable to register buffer object (cudaGraphicsGLRegisterBuffer)."
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }


            if( memory._hostPtr != NULL )
                memory._syncOp |= osgCompute::SYNC_DEVICE;
            else
                memory._syncOp |= osgCompute::SYNC_HOST;

            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool GeometryMemory::sync( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        GeometryObject& memory = *memoryPtr;

        /////////////////
        // SYNC MEMORY //
        /////////////////
        if( mapping & osgCompute::MAP_DEVICE )
        {
            if( !(memory._syncOp & osgCompute::SYNC_DEVICE) )
                return true;

            res = cudaMemcpy( memory._devPtr, memory._hostPtr, getAllElementsSize(), cudaMemcpyHostToDevice );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaMemcpy() to device. "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            memory._syncOp = memory._syncOp ^ osgCompute::SYNC_DEVICE;
            return true;
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            if( !(memory._syncOp & osgCompute::SYNC_HOST) )
                return true;

            if( memory._graphicsResource == NULL )
            {
                osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
                osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
                if( !glBO )
                    return false;

                //////////////
                // SETUP BO //
                //////////////
                // compile buffer object if necessary
                if( glBO->isDirty() )
                    glBO->compileBuffer();


                //////////////////
                // REGISTER PBO //
                //////////////////
                if( NULL == memory._graphicsResource )
                {
                    cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _geomref->getName() << ": unable to register buffer object."
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return false;
                    }
                }
            }

            ////////////////
            // MAP BUFFER //
            ////////////////
            if( NULL == memory._devPtr )
            {

                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }

                size_t memSize;
                res = cudaGraphicsResourceGetMappedPointer (&memory._devPtr, &memSize, memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }
            }

            /////////////////
            // COPY MEMORY //
            /////////////////
            res = cudaMemcpy( memory._hostPtr, memory._devPtr, getAllElementsSize(), cudaMemcpyDeviceToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaMemcpy() to host memory. "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            memory._syncOp = memory._syncOp ^ osgCompute::SYNC_HOST;
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    osgCompute::MemoryObject* GeometryMemory::createObject() const
    {
        return new GeometryObject;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    IndexedGeometryMemory::IndexedGeometryMemory()
        :  GeometryMemory()
    {
        _indicesByteSize = 0;
        // Please note that virtual functions className() and libraryName() are called
        // during observeResource() which will only develop until this class.
        // However if contructor of a subclass calls this function again observeResource
        // will change the className and libraryName of the observed pointer.
        osgCompute::ResourceObserver::instance()->observeResource( *this );
    }

    //------------------------------------------------------------------------------
    IndexedGeometryMemory::~IndexedGeometryMemory()
    {
    }

    //------------------------------------------------------------------------------
    void* IndexedGeometryMemory::map( unsigned int mapping /*= osgCompute::MAP_DEVICE*/, unsigned int offset /*= 0*/, unsigned int hint /*= 0 */ )
    {
        if( (mapping & MAP_INDICES) == MAP_INDICES )
            return mapIndices( mapping, offset, hint );
        else
            return GeometryMemory::map( mapping, offset, hint );
    }

    //------------------------------------------------------------------------------
    void* IndexedGeometryMemory::mapIndices( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint/* = 0*/ )
    {
		if( !_geomref.valid() )
			return NULL;

        if( mapping == osgCompute::UNMAP )
        {
            unmapIndices( hint );
            return NULL;
        }

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        IndexedGeometryObject* memoryPtr = dynamic_cast<IndexedGeometryObject*>( object(true) );
        if( !memoryPtr )
            return NULL;
        IndexedGeometryObject& memory = *memoryPtr;

        //////////////
        // MAP DATA //
        //////////////
        void* ptr = NULL;

        osg::ElementBufferObject* ebo = _geomref->getOrCreateElementBufferObject();
        if( !ebo )
            return NULL;


        memory._idxMapping = mapping;
        bool firstLoad = false;
        bool needsSetup = false;


        if( mapping & osgCompute::MAP_HOST )
        {
            //////////////////////////
            // ALLOCATE HOST-MEMORY //
            //////////////////////////
            if( NULL == memory._hostIdxPtr )
            {
                if( !allocIndices( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            // check if element buffers have changed
            if( memory._lastIdxModifiedCount.size() != ebo->getNumBufferData() )
                needsSetup = true;
            else
                for( unsigned int d=0; d<memory._lastIdxModifiedCount.size(); ++d )
                    if( memory._lastIdxModifiedCount[d] != ebo->getBufferData(d)->getModifiedCount() )
                        needsSetup = true;


            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup && !(memory._syncIdxOp & osgCompute::SYNC_HOST) )
                if( !setupIndices( mapping ) )
                    return NULL;

            /////////////////
            // SYNC STREAM //
            /////////////////
            if( memory._syncIdxOp & osgCompute::SYNC_HOST )
            {
                // map's device ptr if necessary
                if( !syncIndices( mapping ) )
                    return NULL;
            }

            ptr = memory._hostIdxPtr;
        }
        else if( (mapping & osgCompute::MAP_DEVICE) )
        {
            osg::GLBufferObject* glBO = ebo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !glBO )
                return NULL;

            needsSetup = glBO->isDirty();

            ////////////////////////////
            // ALLOCATE DEVICE-MEMORY //
            ////////////////////////////
            // create dynamic texture device memory
            // for each type of mapping
            if( NULL == memory._graphicsIdxResource )
            {
                if( !allocIndices( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            /////////////
            // MAP VBO //
            /////////////
            if( NULL == memory._devIdxPtr )
            {
                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsIdxResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }

                size_t memSize = 0;
                res = cudaGraphicsResourceGetMappedPointer(&memory._devIdxPtr, &memSize, memory._graphicsIdxResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }
            }

            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup )
                if( !setupIndices( mapping ) )
                    return NULL;

            /////////////////
            // SYNC STREAM //
            /////////////////
            if( ( memory._syncIdxOp & osgCompute::SYNC_DEVICE ) && NULL != memory._hostIdxPtr )
                if( !syncIndices( mapping ) )
                    return NULL;

            ptr = memory._devIdxPtr;
        }
        else
        {
            osg::notify(osg::WARN)
                << __FUNCTION__ <<" " << _geomref->getName() << ": Wrong mapping type specified. Use one of the following types: "
                << "HOST_SOURCE_INDICES, HOST_TARGET_INDICES, HOST_INDICES, DEVICE_SOURCE_INDICES, DEVICE_TARGET_INDICES, DEVICE_INDICES."
                << std::endl;

            return NULL;
        }

        if( NULL ==  ptr )
            return NULL;

        if( (mapping & osgCompute::MAP_DEVICE_TARGET) == osgCompute::MAP_DEVICE_TARGET )
            memory._syncIdxOp |= osgCompute::SYNC_HOST;

        if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
            memory._syncIdxOp |= osgCompute::SYNC_DEVICE;

        return &static_cast<char*>(ptr)[offset];
    }

    //------------------------------------------------------------------------------
    void IndexedGeometryMemory::unmap( unsigned int )
    {
        GeometryMemory::unmap();
        unmapIndices();
    }

    //------------------------------------------------------------------------------
    void IndexedGeometryMemory::unmapIndices( unsigned int )
    {
		if( !_geomref.valid() )
			return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        IndexedGeometryObject* memoryPtr = dynamic_cast<IndexedGeometryObject*>( object(false) );
        if( !memoryPtr )
            return;
        IndexedGeometryObject& memory = *memoryPtr;


        if( memory._graphicsIdxResource == NULL )
        {
            // Geometry object will be created during rendering
            // so update the host memory during next mapping
            memory._syncIdxOp |= osgCompute::SYNC_HOST;
        }

        //////////////////
        // UNMAP MEMORY //
        //////////////////
        // Copy host memory to EBO
        if( memory._syncIdxOp & osgCompute::SYNC_DEVICE )
        {
            // Will remove sync flag
            if( NULL == mapIndices( MAP_DEVICE_SOURCE_INDICES, 0 ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during device memory synchronization."
                    << std::endl;

                return;
            }
        }

        // Change current context to render context
        if( memory._devIdxPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsIdxResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            memory._devIdxPtr = NULL;
            memory._idxMapping = osgCompute::UNMAP;
        }
    }

    //------------------------------------------------------------------------------
    bool IndexedGeometryMemory::reset( unsigned int hint /*= 0 */ )
    {
        if( !resetIndices( hint ) )
            return false;

        return GeometryMemory::reset( hint );
    }

    //------------------------------------------------------------------------------
    bool IndexedGeometryMemory::resetIndices( unsigned int  )
    {
		if( !_geomref.valid() )
			return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        IndexedGeometryObject* memoryPtr = dynamic_cast<IndexedGeometryObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        IndexedGeometryObject& memory = *memoryPtr;

        ////////////////////////
        // CLEAR MEMORY FIRST //
        ////////////////////////
        if( memory._hostIdxPtr != NULL )
        {
            if( !memset( memory._hostIdxPtr, 0x0, getIndicesByteSize() ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during memset() for host memory."
                    << std::endl;

                return false;
            }
        }

        // clear device memory
        if( memory._graphicsIdxResource != NULL )
        {
            if( memory._devIdxPtr == NULL )
            {
                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsIdxResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }

                size_t memSize = 0;
                res = cudaGraphicsResourceGetMappedPointer(&memory._devIdxPtr, &memSize, memory._graphicsIdxResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }
            }

            cudaError res = cudaMemset( memory._devIdxPtr, 0x0, getIndicesByteSize() );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaMemset(). "
                    << cudaGetErrorString( res )  <<"."
                    << std::endl;

                return NULL;
            }
        }

        ///////////////////////
        // UNREGISTER BUFFER //
        ///////////////////////
        if( memory._devIdxPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsIdxResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return false;
            }

            memory._devIdxPtr = NULL;
        }

        if( memory._graphicsIdxResource != NULL )
        {
            cudaError_t res = cudaGraphicsUnregisterResource( memory._graphicsIdxResource );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsUnregisterResource()."
                    << cudaGetErrorString(res) << std::endl;
                return false;
            }

            memory._graphicsIdxResource = NULL;
        }


        //////////////////
        // RESET MEMORY //
        //////////////////
        // Reset array data
        memory._lastIdxModifiedCount.clear();
        memory._syncIdxOp = osgCompute::NO_SYNC;

        osg::ElementBufferObject* ebo = _geomref->getOrCreateElementBufferObject();
        if( !ebo )
        {
            osg::notify(osg::FATAL)
                << __FUNCTION__ <<" " << _geomref->getName() << ": no buffer object found."
                << std::endl;

            return false;
        }

        ebo->dirty();

        // Compile element buffer
        osg::GLBufferObject* glBO = ebo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
        if( NULL == glBO )
        {
            osg::notify(osg::FATAL)
                << __FUNCTION__ <<" " << _geomref->getName() << ": no GL buffer object found."
                << std::endl;

            return false;
        }

        glBO->dirty();

        return true;
    }

    //------------------------------------------------------------------------------
    bool IndexedGeometryMemory::supportsMapping( unsigned int mapping, unsigned int ) const
    {
        switch( mapping )
        {
        case osgCompute::UNMAP:
        case osgCompute::MAP_HOST:
        case osgCompute::MAP_HOST_SOURCE:
        case osgCompute::MAP_HOST_TARGET:
        case osgCompute::MAP_DEVICE:
        case osgCompute::MAP_DEVICE_SOURCE:
        case osgCompute::MAP_DEVICE_TARGET:
        case MAP_DEVICE_INDICES:      
        case MAP_DEVICE_TARGET_INDICES:
        case MAP_DEVICE_SOURCE_INDICES:
        case MAP_HOST_INDICES:
        case MAP_HOST_TARGET_INDICES: 
        case MAP_HOST_SOURCE_INDICES:
            return true;
        default:
            return false;
        }
    }

    //------------------------------------------------------------------------------
    unsigned int IndexedGeometryMemory::getIndicesByteSize() const
    {
        if( _indicesByteSize == 0 )
        {
            if( !_geomref.valid() )
                return 0;

            osg::Geometry::DrawElementsList drawElementsList;
            _geomref->getDrawElementsList( drawElementsList );

            if( drawElementsList.size() == 0 )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": no indices defined! setup element array first."
                    << std::endl;

                return 0;
            }

            _indicesByteSize = 0;
            for( unsigned int a=0; a<drawElementsList.size(); ++a )
                _indicesByteSize += drawElementsList[a]->getTotalDataSize();
        }


        return _indicesByteSize;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool IndexedGeometryMemory::setupIndices( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        IndexedGeometryObject* memoryPtr = dynamic_cast<IndexedGeometryObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        IndexedGeometryObject& memory = *memoryPtr;

        //////////////////
        // SETUP MEMORY //
        //////////////////
        osg::ElementBufferObject* ebo = _geomref.get()->getOrCreateElementBufferObject();
        if( mapping & osgCompute::MAP_DEVICE )
        {
            osg::GLBufferObject* glBO = ebo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !glBO->isDirty() )
                return true;

            ////////////////////
            // UNREGISTER VBO //
            ////////////////////
            bool wasRegistered = false;
            if( memory._graphicsIdxResource != NULL )
            {
                wasRegistered = true;
                cudaError res = cudaGraphicsUnregisterResource( memory._graphicsIdxResource );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": unable to unregister buffer object. "
                        << std::endl;

                    return false;
                }
                memory._graphicsIdxResource = NULL;
            }

            ////////////////
            // UPDATE VBO //
            ////////////////
            if( glBO->isDirty() )
                glBO->compileBuffer();

            //////////////////
            // REGISTER VBO //
            //////////////////
            if( wasRegistered )
            {
                cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsIdxResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": unable to register buffer object again."
                        << std::endl;

                    return false;
                }
            }

            memory._syncIdxOp = osgCompute::SYNC_HOST;
        }
        else //  mapping & osgCompute::MAP_HOST
        {
            unsigned char* hostIdxPtr = static_cast<unsigned char*>( memory._hostIdxPtr );

            // Copy buffers into host memory
            if( memory._lastIdxModifiedCount.size() != ebo->getNumBufferData() )
                memory._lastIdxModifiedCount.resize( ebo->getNumBufferData(), UINT_MAX );

            unsigned int curOffset = 0;
            for( unsigned int d=0; d< ebo->getNumBufferData(); ++d )
            {
                osg::BufferData* curData = ebo->getBufferData(d);
                if( !curData )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": invalid buffer data found."
                        << std::endl;

                    return false;
                }

                if( curData->getModifiedCount() != memory._lastIdxModifiedCount[d] )
                {
                    // Copy memory
                    memcpy( &hostIdxPtr[curOffset], curData->getDataPointer(), curData->getTotalDataSize() );

                    // Store last modified value
                    memory._lastIdxModifiedCount[d] = curData->getModifiedCount();
                }
                curOffset += curData->getTotalDataSize();
            }

            memory._syncIdxOp = osgCompute::SYNC_DEVICE;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool IndexedGeometryMemory::allocIndices( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        IndexedGeometryObject* memoryPtr = dynamic_cast<IndexedGeometryObject*>( object(true) );
        if( !memoryPtr )
            return NULL;
        IndexedGeometryObject& memory = *memoryPtr;

        //////////////////////////////
        // ALLOCATE/REGSITER MEMORY //
        //////////////////////////////
        if( mapping & osgCompute::MAP_HOST )
        {
            if( memory._hostIdxPtr != NULL )
                return true;

            memory._hostIdxPtr = malloc( getIndicesByteSize() );
            if( NULL == memory._hostIdxPtr )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during mallocHost()."
                    << std::endl;

                return false;
            }

            if( memory._devIdxPtr != NULL || 
                (memory._syncIdxOp & osgCompute::SYNC_HOST) )
            {
                memory._syncIdxOp |= osgCompute::SYNC_HOST;

                // synchronize host memory with device memory and avoid copying data
                // from buffers in first place
                osg::ElementBufferObject* ebo = _geomref->getOrCreateElementBufferObject();
                for( unsigned int d=0; d< ebo->getNumBufferData(); ++d )
                    memory._lastIdxModifiedCount.push_back( ebo->getBufferData(d)->getModifiedCount() );
            }
            else
            {
                // mark buffers to be copied into the host memory
                osg::ElementBufferObject* ebo = _geomref->getOrCreateElementBufferObject();
                for( unsigned int d=0; d< ebo->getNumBufferData(); ++d )
                    memory._lastIdxModifiedCount.push_back( UINT_MAX );

                memory._syncIdxOp |= osgCompute::SYNC_DEVICE;
            }
            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            osg::ElementBufferObject* ebo = _geomref->getOrCreateElementBufferObject();
            if( !ebo )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": no buffer object found."
                    << std::endl;

                return false;
            }

            // Compile vertex buffer
            osg::GLBufferObject* glBO = ebo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !glBO )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": no GL buffer object found."
                    << std::endl;

                return false;
            }

            if( glBO->isDirty() )
                glBO->compileBuffer();

            // Register vertex buffer 
            cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsIdxResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": unable to register buffer object (cudaGraphicsGLRegisterBuffer)."
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            if( memory._hostIdxPtr != NULL )
                memory._syncIdxOp |= osgCompute::SYNC_DEVICE;
            else
                memory._syncIdxOp |= osgCompute::SYNC_HOST;

            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool IndexedGeometryMemory::syncIndices( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        IndexedGeometryObject* memoryPtr = dynamic_cast<IndexedGeometryObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        IndexedGeometryObject& memory = *memoryPtr;

        /////////////////
        // SYNC MEMORY //
        /////////////////
        if( mapping & osgCompute::MAP_DEVICE )
        {
            if( !(memory._syncIdxOp & osgCompute::SYNC_DEVICE) )
                return true;

            res = cudaMemcpy( memory._devIdxPtr, memory._hostIdxPtr, getIndicesByteSize(), cudaMemcpyHostToDevice );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaMemcpy() to device. "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            memory._syncIdxOp = memory._syncIdxOp ^ osgCompute::SYNC_DEVICE;
            return true;
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            if( !(memory._syncIdxOp & osgCompute::SYNC_HOST) )
                return true;

            if( memory._graphicsIdxResource == NULL )
            {
                osg::ElementBufferObject* ebo = _geomref.get()->getOrCreateElementBufferObject();
                osg::GLBufferObject* glBO = ebo->getOrCreateGLBufferObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
                if( !glBO )
                    return false;

                //////////////
                // SETUP BO //
                //////////////
                // compile buffer object if necessary
                if( glBO->isDirty() )
                    glBO->compileBuffer();

                //////////////////
                // REGISTER PBO //
                //////////////////
                if( NULL == memory._graphicsIdxResource )
                {
                    cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsIdxResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _geomref->getName() << ": unable to register buffer object."
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return false;
                    }
                }
            }

            ////////////////
            // MAP BUFFER //
            ////////////////
            if( NULL == memory._devIdxPtr )
            {
                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsIdxResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }

                size_t memSize;
                res = cudaGraphicsResourceGetMappedPointer (&memory._devIdxPtr, &memSize, memory._graphicsIdxResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }
            }

            /////////////////
            // COPY MEMORY //
            /////////////////
            res = cudaMemcpy( memory._hostIdxPtr, memory._devIdxPtr, getIndicesByteSize(), cudaMemcpyDeviceToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _geomref->getName() << ": error during cudaMemcpy() to host memory. "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            memory._syncIdxOp = memory._syncIdxOp ^ osgCompute::SYNC_HOST;
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    void IndexedGeometryMemory::clear()
    {
        _indicesByteSize = 0;
        osgCompute::GLMemory::clear();
    }

    //------------------------------------------------------------------------------
    osgCompute::MemoryObject* IndexedGeometryMemory::createObject() const
    {
        return new IndexedGeometryObject;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Geometry::Geometry()
        : osg::Geometry(),
		  osgCompute::GLMemoryAdapter()
    {
		GeometryMemory* memory = new GeometryMemory;
		memory->_geomref = this;
		_memory = memory;

        // geometry must use vertex buffer objects
        setUseVertexBufferObjects( true );

    }

    //------------------------------------------------------------------------------
    osgCompute::GLMemory* Geometry::getMemory()
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    const osgCompute::GLMemory* Geometry::getMemory() const
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    void Geometry::addIdentifier( const std::string& identifier )
    {
        _memory->addIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    void Geometry::removeIdentifier( const std::string& identifier )
    {
        _memory->removeIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    bool Geometry::isIdentifiedBy( const std::string& identifier ) const
    {
        return _memory->isIdentifiedBy( identifier );
    }

	//------------------------------------------------------------------------------
	osgCompute::IdentifierSet& Geometry::getIdentifiers()
	{
		return _memory->getIdentifiers();
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& Geometry::getIdentifiers() const
	{
		return _memory->getIdentifiers();
	}

    //------------------------------------------------------------------------------
    void Geometry::applyAsRenderTarget() const
    {
        // Do nothing as geometry cannot be mapped as a render target.
    }

    //------------------------------------------------------------------------------
    void Geometry::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        // Currently we support  a single OpenGL context only. So clear memory every
        // time releaseGLObjects() is called.
        //if( state != NULL && 
        //    osgCompute::GLMemory::getContext() != NULL && 
        //    state->getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->clear();

        osg::Geometry::releaseGLObjects( state );
    }

    ////------------------------------------------------------------------------------
    //void Geometry::resizeGLObjectBuffers( unsigned int maxSize )
    //{
    //    // Currently we support  a single OpenGL context only. So clear memory every
    //    // time releaseGLObjects() is called.
    //    //if( osgCompute::GLMemory::getContext() != NULL )
    //    _memory->clear();

    //    osg::Geometry::resizeGLObjectBuffers( maxSize );
    //}

    //------------------------------------------------------------------------------
    void Geometry::drawImplementation( osg::RenderInfo& renderInfo ) const
    {
        //if( osgCompute::GLMemory::getContext() != NULL &&
        //    renderInfo.getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->unmap(); 

        osg::Geometry::drawImplementation( renderInfo );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Geometry::~Geometry()
    {
		// _memory object is not deleted until this point
		// as reference count is increased in constructor.
        // Do also call releaseObjects()
        _memory->releaseObjects();
        _memory = NULL;
    }
}
