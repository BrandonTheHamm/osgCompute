#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
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
        _lastModifiedCount.clear();
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
    osgCompute::InteropObject* GeometryBuffer::getObject()
    { 
        return _geomref.get(); 
    }

    //------------------------------------------------------------------------------
    void GeometryBuffer::clear()
    {
        osgCompute::Buffer::clear();
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

        return osgCompute::Buffer::init();
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

        GeometryStream* stream = static_cast<GeometryStream*>( lookupStream() );
        if( NULL == stream )
        {
            osg::notify(osg::FATAL)
                << getName() << " [GeometryBuffer::map()]: cannot receive geometry stream."
                << std::endl;

            return NULL;
        }


        void* ptr = mapStream( *stream, mapping, offset );
        if(NULL !=  ptr )
        {
            if( (mapping & osgCompute::MAP_DEVICE_TARGET) == osgCompute::MAP_DEVICE_TARGET )
                stream->_syncHost = true;

            if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
                stream->_syncDevice = true;
        }

        return ptr;
    }

    //------------------------------------------------------------------------------
    void GeometryBuffer::unmap( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return;

        GeometryStream* stream = static_cast<GeometryStream*>( lookupStream() );
        if( NULL == stream )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::GeometryBuffer::map()]: cannot receive geometry stream."
                << std::endl;

            return;
        }

        unmapStream( *stream );
    }

    //------------------------------------------------------------------------------
    bool osgCuda::GeometryBuffer::setMemory( int value, unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int count/* = UINT_MAX*/, unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

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
    bool GeometryBuffer::resetMemory( unsigned int  )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

        GeometryStream* stream = static_cast<GeometryStream*>( lookupStream() );
        if( NULL == stream )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::GeometryBuffer::resetMemory()]: could not receive BufferStream."
                << std::endl;

            return false;
        }

        // reset array data
        stream->_lastModifiedCount.clear();

        // reset host memory
        if( stream->_hostPtr != NULL )
        {
            if( !memset( stream->_hostPtr, 0x0, getByteSize() ) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::resetMemory()]: error during memset() for host memory."
                    << std::endl;

                return false;
            }

            stream->_mapping = osgCompute::MAP_HOST;
            stream->_syncHost = false;
        }

        // reset device memory
        if( stream->_bo != UINT_MAX )
        {
            if( stream->_devPtr == NULL )
            {
                cudaError res = cudaGLMapBufferObject( &stream->_devPtr, stream->_bo );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::GeometryBuffer::resetMemory()]: error during cudaGLMapBufferObject()."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return NULL;
                }
            }

            cudaError res = cudaMemset( stream->_devPtr, 0x0, getByteSize() );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::resetMemory()]: error during cudaMemset() for device memory."
                    << std::endl;

                return false;
            }

            stream->_mapping = osgCompute::MAP_DEVICE;
            stream->_syncDevice = false;
        }

        return true;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void GeometryBuffer::clearLocal()
    {
    }

    //------------------------------------------------------------------------------
    void GeometryBuffer::clear( const osgCompute::Context& context ) const
    {
        if( osgCompute::Context::getAppliedContext() != &context )
            const_cast<osgCompute::Context*>( &context )->apply();

        if( getMapping() != osgCompute::UNMAPPED )
            const_cast<GeometryBuffer*>(this)->unmap();

        osgCompute::Buffer::clear( context );
    }

    //------------------------------------------------------------------------------
    void* GeometryBuffer::mapStream( GeometryStream& stream, unsigned int mapping, unsigned int offset )
    {
        void* ptr = NULL;

        osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
        if( !vbo )
            return NULL;


        stream._mapping = mapping;
        bool firstLoad = false;
        bool needsSetup = false;

        //////////////
        // MAP DATA //
        //////////////
        if( mapping & osgCompute::MAP_HOST )
        {
            //////////////////////////
            // ALLOCATE HOST-MEMORY //
            //////////////////////////
            if( NULL == stream._hostPtr )
            {
                if( !allocStream( mapping, stream ) )
                    return NULL;

                firstLoad = true;
            }


            // check if buffers have changed
            if( stream._lastModifiedCount.size() != vbo->getNumBufferData() )
                needsSetup = true;
            else
                for( unsigned int d=0; d<stream._lastModifiedCount.size(); ++d )
                    if( stream._lastModifiedCount[d] != vbo->getBufferData(d)->getModifiedCount() )
                        needsSetup = true;

            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup && !stream._syncHost )
                if( !setupStream( mapping, stream ) )
                    return NULL;

            /////////////////
            // SYNC STREAM //
            /////////////////
            if( stream._syncHost )
            {
                if( !syncStream( mapping, stream ) )
                    return NULL;
            }

            ptr = stream._hostPtr;
        }
        else if( (mapping & osgCompute::MAP_DEVICE) )
        {
            unsigned int glCtxId = stream._context->getGraphicsContext()->getState()->getContextID();
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( glCtxId );
            if( !glBO )
                return NULL;

            needsSetup = glBO->isDirty();

            ////////////////////////////
            // ALLOCATE DEVICE-MEMORY //
            ////////////////////////////
            // create dynamic texture device memory
            // for each type of mapping
            if( UINT_MAX == stream._bo )
            {
                if( !allocStream( osgCompute::MAP_DEVICE, stream ) )
                    return NULL;

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
                        << getName() << " [osgCuda::GeometryBuffer::mapStream()]: error during cudaGLMapBufferObject(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }
            }

            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup && !stream._syncDevice )
                if( !setupStream( mapping, stream ) )
                    return NULL;

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
                << getName() << " [osgCuda::GeometryBuffer::mapStream()]: Wrong mapping type specified. Use one of the following types: "
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
                    callback->load( ptr, mapping, offset, *this );
                else
                    callback->subload( ptr, mapping, offset, *this );
            }
        }

        stream._mapping = mapping;
        return &static_cast<char*>(ptr)[offset];
    }

    //------------------------------------------------------------------------------
    void GeometryBuffer::unmapStream( GeometryStream& stream )
    {
        // Copy host memory to VBO
        if( stream._syncDevice )
        {
            if( NULL == mapStream( stream, osgCompute::MAP_DEVICE_SOURCE, 0 ) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::unmapStream()]: error during device memory synchronization (mapStream())."
                    << std::endl;

                return;
            }
        }

        // Change current context to render context
        if( stream._devPtr != NULL && stream._bo != UINT_MAX)
        {
            cudaError res = cudaGLUnmapBufferObject( stream._bo );
            if( cudaSuccess != res )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::GeometryBuffer::unmapStream()]: error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            stream._devPtr = NULL;
            stream._mapping = osgCompute::UNMAPPED;
        }
    }

    //------------------------------------------------------------------------------
    void GeometryBuffer::checkMappingWithinDraw( const osgCompute::Context& context )
    {
        if( !context.isConnectedWithGraphicsContext() )
            return;

        GeometryStream* stream = static_cast<GeometryStream*>( lookupStream() );
        if( NULL == stream )
            return;

        if( stream->_bo == UINT_MAX )
        {
            // Geometry object will be created during rendering
            // so update the host memory during next mapping
            stream->_syncHost = true;
        }

        // unmap stream
        if( stream->_mapping != osgCompute::UNMAPPED )
            unmapStream( *stream );
    }

    //------------------------------------------------------------------------------
    bool GeometryBuffer::setupStream( unsigned int mapping, GeometryStream& stream )
    {
        osg::State* state = stream._context->getGraphicsContext()->getState();
        if( state == NULL )
            return false;

        osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
        if( mapping & osgCompute::MAP_DEVICE )
        {
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( state->getContextID() );
            if( !glBO->isDirty() )
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
                        << getName() << " [osgCuda::GeometryBuffer::setupStream()]: unable to unregister buffer object. "
                        << std::endl;

                    return false;
                }
            }
            stream._boRegistered = false;

            ////////////////
            // UPDATE VBO //
            ////////////////
            glBO->compileBuffer();

            //////////////////
            // REGISTER VBO //
            //////////////////
            cudaError_t res = cudaGLRegisterBufferObject( stream._bo );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::setupStream()]: unable to register buffer object again."
                    << std::endl;

                return false;
            }

            stream._boRegistered = true;
            stream._syncHost = true;
        }
        else //  mapping & osgCompute::MAP_HOST
        {
            unsigned char* hostPtr = static_cast<unsigned char*>( stream._hostPtr );

            // copy buffers into host memory
            if( stream._lastModifiedCount.size() != vbo->getNumBufferData() )
                stream._lastModifiedCount.resize( vbo->getNumBufferData(), UINT_MAX );

            unsigned int curOffset = 0;
            for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
            {
                osg::BufferData* curData = vbo->getBufferData(d);
                if( !curData )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::GeometryBuffer::setupStream()]: invalid buffer data found."
                        << std::endl;

                    return false;
                }

                if( curData->getModifiedCount() != stream._lastModifiedCount[d] )
                {
                    // copy memory
                    memcpy( &hostPtr[curOffset], curData->getDataPointer(), curData->getTotalDataSize() );

                    // store last modified count
                    stream._lastModifiedCount[d] = curData->getModifiedCount();
                }
                curOffset += curData->getTotalDataSize();
            }

            stream._syncDevice = true;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool GeometryBuffer::allocStream( unsigned int mapping, GeometryStream& stream )
    {
        if( mapping & osgCompute::MAP_HOST )
        {
            if( stream._hostPtr != NULL )
                return true;

            stream._hostPtr = malloc( getByteSize() );
            if( NULL == stream._hostPtr )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::allocStream()]: error during mallocHost()."
                    << std::endl;

                return false;
            }

            stream._hostPtrAllocated = true;
            if( stream._devPtr != NULL || stream._syncHost )
            {
                stream._syncHost = true;

                // sync host memory with device memory therefore avoid copying data
                // from buffers first
                unsigned int curOffset = 0;
                osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
                for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
                    stream._lastModifiedCount.push_back( vbo->getBufferData(d)->getModifiedCount() );
            }
            else
            {
                // mark buffers to be copied into the host memory
                unsigned int curOffset = 0;
                osg::VertexBufferObject* vbo = _geomref->getOrCreateVertexBufferObject();
                for( unsigned int d=0; d< vbo->getNumBufferData(); ++d )
                    stream._lastModifiedCount.push_back( UINT_MAX );

                stream._syncDevice = true;
            }
            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            if( stream._bo != UINT_MAX )
                return true;

            osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
            osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( stream._context->getGraphicsContext()->getState()->getContextID() );
            if( !glBO )
                return false;

            //////////////
            // SETUP BO //
            //////////////
            // compile buffer object if necessary
            if( glBO->isDirty() )
                glBO->compileBuffer();

            // using vertex buffers
            if( stream._bo == UINT_MAX )
                stream._bo = glBO->getGLObjectID();

            //////////////////
            // REGISTER PBO //
            //////////////////
            if( !stream._boRegistered )
            {
                cudaError_t res = cudaGLRegisterBufferObject( stream._bo );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::GeometryBuffer::allocStream()]: unable to register buffer object."
                        << std::endl;

                    return false;
                }
            }
            stream._boRegistered = true;

            if( stream._hostPtr != NULL )
                stream._syncDevice = true;
            else
                stream._syncHost = true;
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool GeometryBuffer::syncStream( unsigned int mapping, GeometryStream& stream )
    {
        cudaError res;
        if( mapping & osgCompute::MAP_DEVICE )
        {
            res = cudaMemcpy( stream._devPtr, stream._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::syncStream()]: error during cudaMemcpy() to device. "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return false;
            }

            stream._syncDevice = false;
            return true;
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            if( stream._bo == UINT_MAX )
            {
                osg::VertexBufferObject* vbo = _geomref.get()->getOrCreateVertexBufferObject();
                osg::GLBufferObject* glBO = vbo->getOrCreateGLBufferObject( stream._context->getGraphicsContext()->getState()->getContextID() );
                if( !glBO )
                    return false;

                //////////////
                // SETUP BO //
                //////////////
                // compile buffer object if necessary
                if( glBO->isDirty() )
                    glBO->compileBuffer();

                // using vertex buffers
                if( stream._bo == UINT_MAX )
                    stream._bo = glBO->getGLObjectID();

                //////////////////
                // REGISTER PBO //
                //////////////////
                if( !stream._boRegistered )
                {
                    cudaError_t res = cudaGLRegisterBufferObject( stream._bo );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::GeometryBuffer::syncStream()]: unable to register buffer object."
                            << std::endl;

                        return false;
                    }
                }
                stream._boRegistered = true;
            }

            if( NULL == stream._devPtr )
            {
                cudaError res = cudaGLMapBufferObject( reinterpret_cast<void**>(&stream._devPtr), stream._bo );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::GeometryBuffer::mapStream()]: error during cudaGLMapBufferObject(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }
            }

            res = cudaMemcpy( stream._hostPtr, stream._devPtr, getByteSize(), cudaMemcpyDeviceToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::GeometryBuffer::syncStream()]: error during cudaMemcpy() to host memory. "
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
    osgCompute::BufferStream* GeometryBuffer::newStream() const
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
            const osgCompute::Context* curCtx = osgCompute::Context::getContextFromGraphicsContext( state->getContextID() );
            if( NULL != curCtx && NULL != _proxy )
            {
                if( osgCompute::Context::getAppliedContext() != curCtx )
                    const_cast<osgCompute::Context*>(curCtx)->apply();

                if( _proxy->getMapping() != osgCompute::UNMAPPED )
                    _proxy->unmap();

                _proxy->clear( *curCtx );
            }
        }

        osg::Geometry::releaseGLObjects( state );
    }

    //------------------------------------------------------------------------------
    void Geometry::drawImplementation( osg::RenderInfo& renderInfo ) const
    {
        const osgCompute::Context* curCtx = osgCompute::Context::getContextFromGraphicsContext( renderInfo.getState()->getContextID() );
        if( NULL != curCtx && NULL != _proxy )
        {
            if( osgCompute::Context::getAppliedContext() != curCtx )
                const_cast<osgCompute::Context*>(curCtx)->apply();

            _proxy->checkMappingWithinDraw( *curCtx );
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

        _handles.clear();
    }
}
