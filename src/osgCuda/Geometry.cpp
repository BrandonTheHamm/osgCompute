#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
#include <osg/GL>
#include <osg/RenderInfo>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Geometry>

namespace osgCuda
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    GeometryObject::GeometryObject()
        :	osgCompute::MemoryObject(),
        _devPtr(NULL),
        _hostPtr(NULL),
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
                osg::notify(osg::WARN)
                    << "[GeometryObject::~GeometryObject()]: error during cudaGLUnmapBufferObject(). "
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
                <<"[GeometryObject::~GeometryObject()]: error during cudaGraphicsUnregisterResource()."
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
    GeometryBuffer::GeometryBuffer()
        :  osgCompute::InteropMemory()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    GeometryBuffer::~GeometryBuffer()
    {
        clearLocal();

        // free this proxy object
        getObject()->freeProxy();
        // decrease reference count of geometry reference
        _geomref = NULL;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropObject* GeometryBuffer::getObject()
    { 
        return _geomref.get(); 
    }

    //------------------------------------------------------------------------------
    void GeometryBuffer::clear()
    {
        osgCompute::InteropMemory::clear();
        clearLocal();
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
                << getName() << " [osgCuda::GeometryBuffer::init()]: no dimensions defined for geometry! setup vertex array first."
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

        return osgCompute::InteropMemory::init();
    }

    //------------------------------------------------------------------------------
    void* GeometryBuffer::map( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint/* = 0*/ )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return NULL;

        if( mapping == osgCompute::UNMAPPED )
        {
            unmap( hint );
            return NULL;
        }

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object() );
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
                if( !setup( mapping ) )
                    return NULL;

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
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::Resource::getCurrentIdx() );
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
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::GeometryBuffer::map()]: error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }

                size_t memSize = 0;
                res = cudaGraphicsResourceGetMappedPointer(&memory._devPtr, &memSize, memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::GeometryBuffer::map()]: error during cudaGraphicsResourceGetMappedPointer(). "
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
                << getName() << " [osgCuda::GeometryBuffer::map()]: Wrong mapping type specified. Use one of the following types: "
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
    void GeometryBuffer::unmap( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object() );
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
                    << getName() << " [osgCuda::GeometryBuffer::unmap()]: error during device memory synchronization (map())."
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
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::GeometryBuffer::unmap()]: error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            memory._devPtr = NULL;
            memory._mapping = osgCompute::UNMAPPED;
        }
    }

    //------------------------------------------------------------------------------
    bool osgCuda::GeometryBuffer::set( int value, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int count/* = UINT_MAX*/, unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

        // A call to map will setup correct sync flags
        unsigned char* data = static_cast<unsigned char*>( map( mapping ) );
        if( NULL == data )
            return false;

        if( mapping & osgCompute::MAP_HOST_TARGET )
        {
            if( NULL == memset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count ) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::setMemory()]: error during memset() for host."
                    << std::endl;

                unmap();
                return false;
            }
        }
        else if( mapping & osgCompute::MAP_DEVICE_TARGET )
        {
            cudaError res = cudaMemset( &data[offset], value, (count == UINT_MAX)? getByteSize() : count );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::setMemory()]: error during cudaMemset() for device data."
                    << std::endl;

                unmap();
                return false;
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool GeometryBuffer::reset( unsigned int  )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object() );
        if( !memoryPtr )
            return NULL;
        GeometryObject& memory = *memoryPtr;

        ///////////////////////
        // UNREGISTER BUFFER //
        ///////////////////////
        if( memory._devPtr != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::WARN)
                    << "[osgCuda::GeometryBuffer::reset()]: error during cudaGLUnmapBufferObject(). "
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
                    <<"[osgCuda::GeometryBuffer::reset()]: error during cudaGraphicsUnregisterResource()."
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
                << getName() << " [osgCuda::GeometryBuffer::reset()]: no buffer object found."
                << std::endl;

            return false;
        }

        // Compile vertex buffer
        osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::Resource::getCurrentIdx() );
        if( NULL == glBO )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::GeometryBuffer::reset()]: no GL buffer object found."
                << std::endl;

            return false;
        }

        glBO->dirty();


        return true;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void GeometryBuffer::clearLocal()
    {
        // Do not call _geomref = NULL;
    }

    //------------------------------------------------------------------------------
    unsigned int GeometryBuffer::computePitch() const
    {
        return getDimension(0)*getElementSize();
    }

    //------------------------------------------------------------------------------
    bool GeometryBuffer::setup( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object() );
        if( !memoryPtr )
            return NULL;
        GeometryObject& memory = *memoryPtr;


        //////////////////
        // SETUP MEMORY //
        //////////////////
        osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
        if( mapping & osgCompute::MAP_DEVICE )
        {
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( Resource::getCurrentIdx() );
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
                        << getName() << " [osgCuda::GeometryBuffer::setup()]: unable to unregister buffer object. "
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
                        << getName() << " [osgCuda::GeometryBuffer::setup()]: unable to register buffer object again."
                        << std::endl;

                    return false;
                }
            }

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
                        << getName() << " [osgCuda::GeometryBuffer::setup()]: invalid buffer data found."
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

            memory._syncOp = osgCompute::SYNC_DEVICE;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool GeometryBuffer::alloc( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object() );
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

            memory._hostPtr = malloc( getByteSize() );
            if( NULL == memory._hostPtr )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::alloc()]: error during mallocHost()."
                    << std::endl;

                return false;
            }

            if( memory._devPtr != NULL || 
                (memory._syncOp & osgCompute::SYNC_HOST) )
            {
                memory._syncOp |= osgCompute::SYNC_HOST;

                // synchronize host memory with device memory and avoid copying data
                // from buffers in first place
                osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
                for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
                    memory._lastModifiedCount.push_back( vbo->getBufferData(d)->getModifiedCount() );
            }
            else
            {
                // mark buffers to be copied into the host memory
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
                    << getName() << " [osgCuda::GeometryBuffer::alloc()]: no buffer object found."
                    << std::endl;

                return false;
            }

            // Compile vertex buffer
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::Resource::getCurrentIdx() );
            if( !glBO )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::alloc()]: no GL buffer object found."
                    << std::endl;

                return false;
            }

            if( glBO->isDirty() )
                glBO->compileBuffer();

            // Register vertex buffer 
            cudaError res = cudaGraphicsGLRegisterBuffer ( &memory._graphicsResource, glBO->getGLObjectID(), cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::alloc()]: unable to register buffer object (cudaGraphicsGLRegisterBuffer)."
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
    bool GeometryBuffer::sync( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        GeometryObject* memoryPtr = dynamic_cast<GeometryObject*>( object() );
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

            res = cudaMemcpy( memory._devPtr, memory._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::syncStream()]: error during cudaMemcpy() to device. "
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
                osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( osgCompute::Resource::getCurrentIdx() );
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
                            << getName() << " [osgCuda::GeometryBuffer::sync()]: unable to register buffer object."
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
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::GeometryBuffer::sync()]: error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }

                size_t memSize;
                res = cudaGraphicsResourceGetMappedPointer (&memory._devPtr, &memSize, memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::GeometryBuffer::sync()]: error during cudaGraphicsResourceGetMappedPointer(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }
            }

            /////////////////
            // COPY MEMORY //
            /////////////////
            res = cudaMemcpy( memory._hostPtr, memory._devPtr, getByteSize(), cudaMemcpyDeviceToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::syncStream()]: error during cudaMemcpy() to host memory. "
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
    osgCompute::MemoryObject* GeometryBuffer::createObject() const
    {
        return new GeometryObject;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Geometry::Geometry()
        : osg::Geometry(),
        _proxy(NULL)
    {
        clearLocal();
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
    osgCompute::InteropMemory* Geometry::getMemory()
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    const osgCompute::InteropMemory* Geometry::getMemory() const
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* Geometry::getOrCreateMemory()
    {
        // create proxy buffer on demand
        if( _proxy == NULL )
        {
            _proxy = new GeometryBuffer;
            _proxy->_geomref = this;
            _proxy->setIdentifiers( _identifiers );
            _identifiers.clear();
            if( !_proxy->init() )
            {
                _proxy->unref();
                _proxy = NULL;
            }
        }

        return _proxy;
    }

    //------------------------------------------------------------------------------
    void Geometry::addIdentifier( const std::string& identifier )
    {
        if( _proxy != NULL )
        {
            _proxy->addIdentifier( identifier );
        }
        else
        {
            if( !isAddressedByIdentifier(identifier) )
                _identifiers.insert( identifier );
        }
    }

    //------------------------------------------------------------------------------
    void Geometry::removeIdentifier( const std::string& identifier )
    {
        if( _proxy != NULL )
        {
            _proxy->removeIdentifier( identifier );
        }
        else
        {
            osgCompute::IdentifierSetItr itr = _identifiers.find( identifier );
            if( itr != _identifiers.end() )
                _identifiers.erase( itr );

        }
    }

    //------------------------------------------------------------------------------
    bool Geometry::isAddressedByIdentifier( const std::string& identifier ) const
    {
        if( _proxy != NULL )
        {
            return _proxy->isAddressedByIdentifier( identifier );
        }
        else
        {
            osgCompute::IdentifierSetCnstItr itr = _identifiers.find( identifier );
            if( itr == _identifiers.end() )
                return false;

            return true;
        }
    }

    //------------------------------------------------------------------------------
    void Geometry::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        _proxy->clearCurrent();
        osg::Geometry::releaseGLObjects( state );
    }

    //------------------------------------------------------------------------------
    void Geometry::drawImplementation( osg::RenderInfo& renderInfo ) const
    {
        _proxy->unmap();
        osg::Geometry::drawImplementation( renderInfo );
    }

    //------------------------------------------------------------------------------
    void Geometry::freeProxy()
    {
        // attach identifiers
        _identifiers = _proxy->getIdentifiers();
        // proxy is now deleted
        _proxy = NULL;
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
                << getName() << " [osgCuda::Geometry::destructor()]: proxy is still valid!!!."
                << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void Geometry::clearLocal()
    {
        _clear = true;
        if( NULL != _proxy )
            _proxy->clear();

        _identifiers.clear();
    }
}
