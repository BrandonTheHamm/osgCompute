#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
#include <cuda_runtime.h>
#include <osg/Notify>
#include <osgCuda/Array>

namespace osgCuda
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ArrayObject::ArrayObject()
        :   osgCompute::MemoryObject(),
        _devArray(NULL),
        _hostPtr(NULL),
        _modifyCount(UINT_MAX)
    {
    }

    //------------------------------------------------------------------------------
    ArrayObject::~ArrayObject()
    {
        if( NULL != _devArray )
        {
            cudaError_t res = cudaFreeArray( _devArray );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    <<"ArrayObject::~ArrayObject(): error during cudaFreeArray(). "
                    <<cudaGetErrorString(res)<<std::endl;
            }
        }

        if( NULL != _hostPtr)
            free( _hostPtr );
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Array::Array()
        : osgCompute::Memory()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Array::clear()
    {
        clearLocal();
        osgCompute::Memory::clear();
    }

    //------------------------------------------------------------------------------
    bool Array::init()
    {
        if( getNumDimensions() > 3 )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::Array::init()] \""<<getName()<<"\": the maximum dimension allowed is 3."
                << std::endl;

            clear();
            return false;
        }

        unsigned int numElements = 1;
        for( unsigned int d=0; d<getNumDimensions(); ++d )
            numElements *= getDimension( d );

        unsigned int byteSize = numElements * getElementSize();

        // check stream data
        if( _image.valid() )
        {
            if( _image->getNumMipmapLevels() > 1 )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::init()] \""<<getName()<<"\": image \""
                    << _image->getName() << "\" uses MipMaps which are currently"
                    << "not supported."
                    << std::endl;

                clear();
                return false;
            }

            if( _image->getTotalSizeInBytes() != byteSize )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::init()] \""<<getName()<<"\": size of image \""
                    << _image->getName() << "\" does not match the array size."
                    << std::endl;

                clear();
                return false;
            }
        }

        if( _array.valid() )
        {
            if( _array->getTotalDataSize() != byteSize )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::init()] \""<<getName()<<"\": size of array \""
                    << _array->getName() << "\" is wrong."
                    << std::endl;

                clear();
                return false;
            }
        }

        return osgCompute::Memory::init();
    }

    //------------------------------------------------------------------------------
    void* Array::map( unsigned int mapping/*= osgCompute::MAP_DEVICE*/, unsigned int offset/*= 0*/, unsigned int hint )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return NULL;

        if( (mapping & osgCompute::MAP_DEVICE) == osgCompute::MAP_DEVICE_TARGET )
        {

            osg::notify(osg::WARN)
                << getName() << " [osgCuda::Array::map()] \""<<getName()<<"\": you cannot write into an array on the device. Use one of the following: "
                << "DEVICE_SOURCE, DEVICE."
                << std::endl;

            return NULL;
        }

        if( mapping == osgCompute::UNMAPPED )
        {
            unmap( hint );
            return NULL;
        }

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        ArrayObject* memoryPtr = dynamic_cast<ArrayObject*>( object() );
        if( !memoryPtr )
            return NULL;
        ArrayObject& memory = *memoryPtr;

        /////////////////////////////
        // CHECK FOR MODIFICATIONS //
        /////////////////////////////
        bool needsSetup = false;
        if( (_image.valid() && _image->getModifiedCount() != memory._modifyCount ) ||
            (_array.valid() && _array->getModifiedCount() != memory._modifyCount ) )
            needsSetup = true;

        memory._mapping = mapping;
        bool firstLoad = false;

        //////////////
        // MAP DATA //
        //////////////
        void* ptr = NULL;
        if( (memory._mapping & osgCompute::MAP_HOST) )
        {
            if( NULL == memory._hostPtr )
            {
                // allocate host-memory 
                if( !alloc( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            // setup stream 
            if( needsSetup )
                if( !setup( mapping ) )
                    return NULL;

            // sync stream 
            if( (memory._syncOp & osgCompute::SYNC_HOST) && NULL != memory._devArray )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._hostPtr;
        }
        else if( (memory._mapping & osgCompute::MAP_DEVICE) )
        {
            if( NULL == memory._devArray )
            {
                // allocate device-memory 
                if( !alloc( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            // setup stream 
            if( needsSetup )
                if( !setup( mapping ) )
                    return NULL;

            // sync stream 
            if( (memory._syncOp & osgCompute::SYNC_DEVICE ) && NULL != memory._hostPtr )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._devArray;
        }
        else
        {
            osg::notify(osg::WARN)
                << getName() << " [osgCuda::Array::map()]: Wrong mapping type specified. Use one of the following types: "
                << "HOST_SOURCE, HOST_TARGET, HOST, DEVICE_SOURCE, DEVICE."
                << std::endl;

            return NULL;
        }

        if( NULL == ptr )
            return NULL; // Return if something failed

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

        if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
            memory._syncOp = osgCompute::SYNC_DEVICE;

        return &static_cast<char*>(ptr)[offset];
    }

    //------------------------------------------------------------------------------
    void Array::unmap( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        ArrayObject* memoryPtr = dynamic_cast<ArrayObject*>( object() );
        if( !memoryPtr )
            return;
        ArrayObject& memory = *memoryPtr;

        ////////////////
        // SETUP FLAG //
        ////////////////
        memory._mapping = osgCompute::UNMAPPED;
    }

    //------------------------------------------------------------------------------
    bool Array::reset( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        ArrayObject* memoryPtr = dynamic_cast<ArrayObject*>( object() );
        if( !memoryPtr )
            return false;
        ArrayObject& memory = *memoryPtr;

        //////////////////
        // RESET MEMORY //
        //////////////////
        // reset array or image data
        memory._modifyCount = UINT_MAX;
        memory._syncOp = osgCompute::NO_SYNC;

        // clear host memory
        if( memory._hostPtr != NULL )
        {
            if( !memset( memory._hostPtr, 0x0, getByteSize() ) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::reset()] \""<<getName()<<"\": error during memset() for host memory."
                    << std::endl;

                unmap();
                return false;
            }
        }

        if( memory._devArray != NULL )
        {
            // currently their is now cudaArrayMemset function, so we have to setup the
            // memory with cudaMemcpy operations
            void* hostData = malloc( getByteSize() );
            if( hostData == NULL )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Array::reset()] \""<<getName()<<"\": malloc failed before cudaMemset."
                    << std::endl;

                unmap();
                free( hostData );
                return false;
            }
            memset( hostData, 0x0, getByteSize() );

            // copy data
            if( getNumDimensions() == 3 )
            {
                cudaMemcpy3DParms memCpyParams = {0};
                memCpyParams.dstArray = memory._devArray;
                memCpyParams.kind = cudaMemcpyHostToDevice;
                memCpyParams.srcPtr = make_cudaPitchedPtr((void*)hostData, getDimension(0)*getElementSize(), getDimension(0), getDimension(1));

                cudaExtent arrayExtent = {0};
                arrayExtent.width = getDimension(0);
                arrayExtent.height = getDimension(1);
                arrayExtent.depth = getDimension(2);

                memCpyParams.extent = arrayExtent;

                cudaError res = cudaMemcpy3D( &memCpyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::reset()] \""<<getName()<<"\": cudaMemcpy3D() failed."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    free( hostData );
                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                cudaError res = cudaMemcpy2DToArray( 
                                        memory._devArray, 0, 0, 
                                        hostData, memory._pitch, getDimension(0)*getElementSize(), getDimension(1), 
                                        cudaMemcpyHostToDevice );  
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::reset()] \""<<getName()<<"\": cudaMemcpy2DToArray() failed."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    free( hostData );
                    return false;
                }
            }
            else
            {
                cudaError res = cudaMemcpyToArray( 
                                    memory._devArray, 0, 0, 
                                    hostData, getByteSize(), 
                                    cudaMemcpyHostToDevice);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::reset()] \""<<getName()<<"\": cudaMemcpyToArray() failed."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    free( hostData );
                    return false;
                }
            }

            free( hostData );
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool Array::setup( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        ArrayObject* memoryPtr = dynamic_cast<ArrayObject*>( object() );
        if( !memoryPtr )
            return false;
        ArrayObject& memory = *memoryPtr;

        //////////////////
        // SETUP MEMORY //
        //////////////////
        if( mapping & osgCompute::MAP_DEVICE )
        {
            const void* data = NULL;
            if( _image.valid() )
            {
                data = _image->data();
            }

            if( _array.valid() )
            {
                data = reinterpret_cast<const void*>(_array->getDataPointer());
            }

            if( data == NULL )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::setup()] \""<<getName()<<"\": Cannot receive valid data pointer."
                    << std::endl;

                return false;
            }

            if( getNumDimensions() == 3 )
            {
                cudaMemcpy3DParms memCpyParams = {0};
                memCpyParams.dstArray = memory._devArray;
                memCpyParams.kind = cudaMemcpyHostToDevice;
                memCpyParams.srcPtr = make_cudaPitchedPtr((void*)data, getDimension(0)*getElementSize(), getDimension(0), getDimension(1));

                cudaExtent arrayExtent = {0};
                arrayExtent.width = getDimension(0);
                arrayExtent.height = getDimension(1);
                arrayExtent.depth = getDimension(2);

                memCpyParams.extent = arrayExtent;

                res = cudaMemcpy3D( &memCpyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::setup()] \""<<getName()<<"\": cudaMemcpy3D() failed."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                cudaError res = cudaMemcpy2DToArray( memory._devArray, 0, 0, data, getDimension(0)*getElementSize(), getDimension(0)*getElementSize(), getDimension(1), cudaMemcpyHostToDevice );  
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::set()] \""<<getName()<<"\": cudaMemcpy2DToArray() failed."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                res = cudaMemcpyToArray(memory._devArray, 0, 0, data, getByteSize(), cudaMemcpyHostToDevice);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::setup()] \""<<getName()<<"\": cudaMemcpyToArray() failed."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return false;
                }
            }


            // host must be synchronized
            // because device memory has been modified
            memory._syncOp = osgCompute::SYNC_HOST;
            memory._modifyCount = _image.valid()? _image->getModifiedCount() : _array->getModifiedCount();
            return true;
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            const void* data = NULL;
            if( _image.valid() )
            {
                data = _image->data();
            }

            if( _array.valid() )
            {
                data = reinterpret_cast<const void*>( _array->getDataPointer() );
            }

            if( data == NULL )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::setup()] \""<<getName()<<"\": cannot receive valid data pointer."
                    << std::endl;

                return false;
            }

            res = cudaMemcpy( memory._hostPtr, data, getByteSize(), cudaMemcpyHostToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::setup()}: error during cudaMemcpy()."
                    << " " << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            // device must be synchronized
            // because host memory has been modified
            memory._syncOp = osgCompute::SYNC_DEVICE;
            memory._modifyCount = _image.valid()? _image->getModifiedCount() : _array->getModifiedCount();
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool Array::alloc( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        ArrayObject* memoryPtr = dynamic_cast<ArrayObject*>( object() );
        if( !memoryPtr )
            return false;
        ArrayObject& memory = *memoryPtr;

        //////////////////
        // ALLOC MEMORY //
        //////////////////
        if( mapping & osgCompute::MAP_HOST )
        {
            if( memory._hostPtr != NULL )
                return true;

            memory._hostPtr = malloc( getByteSize() );
            if( NULL == memory._hostPtr )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::allocStream()] \""<<getName()<<"\": something goes wrong within mallocHostMemory()."
                    << std::endl;

                return false;
            }

            memory._pitch = getDimension(0) * getElementSize();
            if( memory._devArray != NULL )
                memory._syncOp |= osgCompute::SYNC_HOST;

            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            if( memory._devArray != NULL )
                return true;

            const cudaChannelFormatDesc& desc = getChannelFormatDesc();
            if( desc.x == INT_MAX && desc.y == INT_MAX && desc.z == INT_MAX )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::allocStream()] \""<<getName()<<"\": no valid ChannelFormatDesc found."
                    << std::endl;

                return false;
            }

            if( getNumDimensions() == 3 )
            {
                cudaExtent extent;
                extent.width = getDimension(0);
                extent.height = (getDimension(1) <= 1)? 0 : getDimension(1);
                extent.depth = (getDimension(2) <= 1)? 0 : getDimension(2);

                // allocate memory
                cudaError res = cudaMalloc3DArray( &memory._devArray, &desc, extent );
                if( cudaSuccess != res || NULL ==  memory._devArray )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::allocStream()] \""<<getName()<<"\": something goes wrong within cudaMalloc3DArray()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }

                memory._pitch = getDimension(0) * getElementSize();
            }
            else if( getNumDimensions() == 2 )
            {
                cudaError res = cudaMallocArray( &memory._devArray, &desc, getDimension(0), (getDimension(1) <= 1)? 0 : getDimension(1) );
                if( cudaSuccess != res || NULL ==  memory._devArray )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::allocStream()] \""<<getName()<<"\": something goes wrong within mallocDevice2DArray()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }

                memory._pitch = getDimension(0) * getElementSize();
            }
            else
            {
                cudaError res = cudaMallocArray( &memory._devArray, &desc, getDimension(0), 1 );
                if( cudaSuccess != res || NULL ==  memory._devArray )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::allocStream()] \""<<getName()<<"\": something goes wrong within mallocDeviceArray()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }

                memory._pitch = getDimension(0) * getElementSize();
            }

            if( memory._hostPtr != NULL )
                memory._syncOp |= osgCompute::SYNC_DEVICE;

            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool Array::sync( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        ArrayObject* memoryPtr = dynamic_cast<ArrayObject*>( object() );
        if( !memoryPtr )
            return false;
        ArrayObject& memory = *memoryPtr;

        //////////////////
        // ALLOC MEMORY //
        //////////////////
        if( mapping & osgCompute::MAP_DEVICE )
        {
            if( !(memory._syncOp & osgCompute::SYNC_DEVICE) )
                return true;

            if( getNumDimensions() == 1 )
            {
                res = cudaMemcpyToArray( memory._devArray, 0, 0, memory._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::sync()] \""<<getName()<<"\": error during cudaMemcpyToArray() to device memory."
                        << " " << cudaGetErrorString( res )<<"."
                        << std::endl;
                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                res = cudaMemcpy2DToArray( memory._devArray,
                    0, 0,
                    memory._hostPtr,
                    memory._pitch,
                    getDimension(0) * getElementSize(),
                    getDimension(1),
                    cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::sync()] \""<<getName()<<"\": error during cudaMemcpy2DToArray() to device memory."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                cudaPitchedPtr pitchPtr = {0};
                pitchPtr.pitch = memory._pitch;
                pitchPtr.ptr = (void*)memory._hostPtr;
                pitchPtr.xsize = getDimension(0);
                pitchPtr.ysize = getDimension(1);

                cudaExtent extent = {0};
                extent.width = getDimension(0);
                extent.height = getDimension(1);
                extent.depth = getDimension(2);

                cudaMemcpy3DParms copyParams = {0};
                copyParams.srcPtr = pitchPtr;
                copyParams.dstArray = memory._devArray;
                copyParams.extent = extent;
                copyParams.kind = cudaMemcpyHostToDevice;

                res = cudaMemcpy3D( &copyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::sync()] \""<<getName()<<"\": error during cudaMemcpy3D() to device memory."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return false;
                }
            }

            memory._syncOp = memory._syncOp ^ osgCompute::SYNC_DEVICE;
            return true;
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            if( !(memory._syncOp & osgCompute::SYNC_HOST) )
                return true;

            if( getNumDimensions() == 3 )
            {
                cudaPitchedPtr pitchPtr = {0};
                pitchPtr.pitch = memory._pitch;
                pitchPtr.ptr = (void*)memory._hostPtr;
                pitchPtr.xsize = getDimension(0);
                pitchPtr.ysize = getDimension(1);

                cudaExtent extent = {0};
                extent.width = getDimension(0);
                extent.height = getDimension(1);
                extent.depth = getDimension(2);

                cudaMemcpy3DParms copyParams = {0};
                copyParams.srcArray = memory._devArray;
                copyParams.dstPtr = pitchPtr;
                copyParams.extent = extent;
                copyParams.kind = cudaMemcpyDeviceToHost;

                res = cudaMemcpy3D( &copyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::sync()] \""<<getName()<<"\": error during cudaMemcpy3D() to host memory."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;

                }
            }
            else if( getNumDimensions() == 2 )
            {
                res = cudaMemcpy2DFromArray(
                    memory._hostPtr,
                    memory._pitch,
                    memory._devArray,
                    0, 0,
                    getDimension(0) * getElementSize(),
                    getDimension(1),
                    cudaMemcpyDeviceToHost );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::sync()] \""<<getName()<<"\": error during cudaMemcpy2DFromArray() to host memory."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                res = cudaMemcpyFromArray( memory._hostPtr, memory._devArray, 0, 0, getByteSize(), cudaMemcpyDeviceToHost );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Array::sync()] \""<<getName()<<"\":  error during cudaMemcpyFromArray() to host memory."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }

            memory._syncOp = memory._syncOp ^ osgCompute::SYNC_HOST;
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    void Array::setImage( osg::Image* image )
    {
        if( !osgCompute::Resource::isClear() && NULL != image)
        {
            if( image->getNumMipmapLevels() > 1 )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::setImage()] \""<<getName()<<"\": image \""
                    << image->getName() << "\" uses MipMaps which are currently"
                    << "not supported."
                    << std::endl;

                return;
            }

            if( image->getTotalSizeInBytes() != getByteSize() )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::setImage()] \""<<getName()<<"\": size of image \""
                    << image->getName() << "\" is wrong."
                    << std::endl;

                return;
            }
        }

        _image = image;
        _array = NULL;
        resetModifiedCounts();
    }

    //------------------------------------------------------------------------------
    osg::Image* Array::getImage()
    {
        return _image.get();
    }

    //------------------------------------------------------------------------------
    const osg::Image* Array::getImage() const
    {
        return _image.get();
    }

    //------------------------------------------------------------------------------
    void Array::setArray( osg::Array* array )
    {
        if( !osgCompute::Resource::isClear() && array != NULL )
        {
            if( array->getTotalDataSize() != getByteSize() )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::Array::setArray()] \""<<getName()<<"\": size of array \""
                    << array->getName() << "\" does not match with the array size."
                    << std::endl;

                return;
            }
        }

        _array = array;
        _image = NULL;
        resetModifiedCounts();
    }

    //------------------------------------------------------------------------------
    osg::Array* Array::getArray()
    {
        return _array.get();
    }

    //------------------------------------------------------------------------------
    const osg::Array* Array::getArray() const
    {
        return _array.get();
    }


    //------------------------------------------------------------------------------
    cudaChannelFormatDesc& Array::getChannelFormatDesc()
    {
        return _channelFormatDesc;
    }

    //------------------------------------------------------------------------------
    const cudaChannelFormatDesc& Array::getChannelFormatDesc() const
    {
        return _channelFormatDesc;
    }

    //------------------------------------------------------------------------------
    void Array::setChannelFormatDesc(cudaChannelFormatDesc& channelFormatDesc)
    {
        if( !osgCompute::Resource::isClear() )
            return;

        _channelFormatDesc = channelFormatDesc;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Array::clearLocal()
    {
        _image = NULL;
        _array = NULL;
        memset( &_channelFormatDesc, INT_MAX, sizeof(cudaChannelFormatDesc) );
    }

    //------------------------------------------------------------------------------
    unsigned int Array::computePitch() const
    {
        // Proof paramters
        if( getDimension(0) == 0 || getElementSize() == 0 ) 
            return 0;

        int device;
        cudaGetDevice( &device );
        cudaDeviceProp devProp;
        cudaGetDeviceProperties( &devProp, device );

        unsigned int remainingAlignmentBytes = (getDimension(0)*getElementSize()) % devProp.textureAlignment;
        if( remainingAlignmentBytes != 0 )
            return (getDimension(0)*getElementSize()) + (devProp.textureAlignment-remainingAlignmentBytes);
        else
            return (getDimension(0)*getElementSize()); // no additional bytes required.
    }

    //------------------------------------------------------------------------------
    osgCompute::MemoryObject* Array::createObject() const
    {
        return new ArrayObject;
    }


    //------------------------------------------------------------------------------
    void Array::resetModifiedCounts() const
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        const ArrayObject* memoryPtr = dynamic_cast<const ArrayObject*>( object() );
        if( !memoryPtr )
            return;
        ArrayObject& memory = const_cast<ArrayObject&>(*memoryPtr);

        //////////////////////////
        // RESET MODIFIED COUNT //
        //////////////////////////
        memory._modifyCount = UINT_MAX;
    }
}
