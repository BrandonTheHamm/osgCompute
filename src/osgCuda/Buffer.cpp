#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
#include <cuda_runtime.h>
#include <driver_types.h>
#include <osg/Notify>
#include <osgCuda/Buffer>

namespace osgCuda
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    BufferObject::BufferObject()
        :   osgCompute::MemoryObject(),
        _devPtr(NULL),
        _hostPtr(NULL),
        _modifyCount(UINT_MAX)
    {
    }

    //------------------------------------------------------------------------------
    BufferObject::~BufferObject()
    {
        if( NULL != _devPtr)
        {
            cudaError res = cudaFree( _devPtr );
            if( res != cudaSuccess )
            {
                osg::notify(osg::WARN)
                    <<"BufferObject::~BufferObject(): error during cudaFree(). "
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
    Buffer::Buffer()
        : osgCompute::Memory()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Buffer::clear()
    {
        clearLocal();
        osgCompute::Memory::clear();
    }

    //------------------------------------------------------------------------------
    bool Buffer::init()
    {
        unsigned int numElements = 1;
        for( unsigned int d=0; d<getNumDimensions(); ++d )
            numElements *= getDimension( d );

        unsigned int byteSize = numElements * getElementSize();

        // check stream data
        if( _image.valid() )
        {
            if( _image->getNumMipmapLevels() > 1 )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::init()] \""<<getName()<<"\": Image \""
                    << _image->getName() << "\" uses MipMaps which are currently"
                    << "not supported."
                    << std::endl;

                clear();
                return false;
            }

            if( _image->getTotalSizeInBytes() != byteSize )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::init()] \""<<getName()<<"\": size of image \""
                    << _image->getName() << "\" is wrong."
                    << std::endl;

                clear();
                return false;
            }
        }

        if( _array.valid() )
        {
            if( _array->getTotalDataSize() != byteSize )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::init()] \""<<getName()<<"\": size of array \""
                    << _array->getName() << "\" is wrong."
                    << std::endl;

                clear();
                return false;
            }
        }

        return osgCompute::Memory::init();
    }

    //------------------------------------------------------------------------------
    void* Buffer::map( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return NULL;

        if( mapping == osgCompute::UNMAP )
        {
            unmap( hint );
            return NULL;
        }

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        BufferObject* memoryPtr = dynamic_cast<BufferObject*>( object() );
        if( !memoryPtr )
            return NULL;
        BufferObject& memory = *memoryPtr;

        /////////////////////////////
        // CHECK FOR MODIFICATIONS //
        /////////////////////////////
        bool firstLoad = false;
        // image or array has changed
        bool needsSetup = false;
        if( (_image.valid() && _image->getModifiedCount() != memory._modifyCount ) ||
            (_array.valid() && _array->getModifiedCount() != memory._modifyCount ) )
            needsSetup = true;

        // current mapping
        memory._mapping = mapping;

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
            if( (memory._syncOp & osgCompute::SYNC_HOST) && 
                NULL != memory._devPtr )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._hostPtr;
        }
        else if( (memory._mapping & osgCompute::MAP_DEVICE) )
        {
            if( NULL == memory._devPtr )
            {
                // allocate device-memory 
                if( !alloc( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            // setup stream 
            if( needsSetup && !(memory._syncOp & osgCompute::SYNC_DEVICE) )
                if( !setup( mapping ) )
                    return NULL;

            // sync stream 
            if( (memory._syncOp & osgCompute::SYNC_DEVICE) && 
                NULL != memory._hostPtr )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._devPtr;
        }
        else
        {
            osg::notify(osg::WARN)
                << getName() << " [osgCuda::Buffer::map()] \""<<getName()<<"\": Wrong mapping. Use one of the following: "
                << "HOST_SOURCE, HOST_TARGET, HOST, DEVICE_SOURCE, DEVICE_TARGET, DEVICE."
                << std::endl;

            return NULL;
        }

        // return if something failed
        if( NULL ==  ptr )
            return NULL; 

        //////////////////
        // LOAD/SUBLOAD //
        //////////////////
        if( getSubloadCallback() )
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

        // check sync
        if( (mapping & osgCompute::MAP_DEVICE_TARGET) == osgCompute::MAP_DEVICE_TARGET )
            memory._syncOp |= osgCompute::SYNC_HOST;

        if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
            memory._syncOp |= osgCompute::SYNC_DEVICE;

        return &static_cast<char*>(ptr)[offset];
    }

    //------------------------------------------------------------------------------
    void Buffer::unmap( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        BufferObject* memoryPtr = dynamic_cast<BufferObject*>( object() );
        if( !memoryPtr )
            return;
        BufferObject& memory = *memoryPtr;

        ////////////////
        // SETUP FLAG //
        ////////////////
        memory._mapping = osgCompute::UNMAP;
    }

    //------------------------------------------------------------------------------
    bool Buffer::reset( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        BufferObject* memoryPtr = dynamic_cast<BufferObject*>( object() );
        if( !memoryPtr )
            return false;
        BufferObject& memory = *memoryPtr;

        // reset memory from array/image data 
        // during next call of map()
        memory._modifyCount = UINT_MAX;
        memory._syncOp = osgCompute::NO_SYNC;

        // clear host memory
        if( memory._hostPtr != NULL )
        {
            if( !memset( memory._hostPtr, 0x0, getByteSize() ) )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::reset()] \"" << getName() << "\": error during memset() for host memory."
                    << std::endl;

                return false;
            }
        }

        // clear device memory
        if( memory._devPtr != NULL )
        {
            cudaError res;
            if( getNumDimensions() == 3 )
            {
                cudaPitchedPtr pitchedPtr = make_cudaPitchedPtr( memory._devPtr, memory._pitch, getDimension(0)*getElementSize(), getDimension(1) );
                cudaExtent extent = make_cudaExtent( getPitch(), getDimension(1), getDimension(2) );
                res = cudaMemset3D( pitchedPtr, 0x0, extent );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::reset()] \"" << getName() << "\": error during cudaMemset3D() for device memory."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    unmap();
                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                res = cudaMemset2D( memory._devPtr, memory._pitch, 0x0, getDimension(0)*getElementSize(), getDimension(1) );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::reset()] \"" << getName() << "\": error during cudaMemset2D() for device memory."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    unmap();
                    return false;
                }
            }
            else
            {
                res = cudaMemset( memory._devPtr, 0x0, getByteSize() );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::reset()] \"" << getName() << "\": error during cudaMemset() for device memory."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    unmap();
                    return false;
                }
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool Buffer::isMappingAllowed( unsigned int mapping, unsigned int ) const
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

    //------------------------------------------------------------------------------
    bool Buffer::setup( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        BufferObject* memoryPtr = dynamic_cast<BufferObject*>( object() );
        if( !memoryPtr )
            return false;
        BufferObject& memory = *memoryPtr;

        //////////////////
        // SETUP MEMORY //
        //////////////////
        if( mapping & osgCompute::MAP_DEVICE )
        {
            void* data = NULL;
            if( _image.valid() )
            {
                data = _image->data();
            }

            if( _array.valid() )
            {
                data = const_cast<GLvoid*>( _array->getDataPointer() );
            }

            if( data == NULL )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::setup()] \""<<getName()<<"\": cannot receive valid data pointer."
                    << std::endl;

                return false;
            }


            if( getNumDimensions() == 3 )
            {
                cudaMemcpy3DParms memcpyParams = {0};
                memcpyParams.dstPtr = make_cudaPitchedPtr( memory._devPtr, memory._pitch, getDimension(0), getDimension(1) );
                memcpyParams.srcPtr = make_cudaPitchedPtr( data, getDimension(0) * getElementSize(), getDimension(0), getDimension(1) );
                memcpyParams.extent = make_cudaExtent( getDimension(0) * getElementSize(), getDimension(1), getDimension(2) );
                memcpyParams.kind = cudaMemcpyHostToDevice;

                res = cudaMemcpy3D( &memcpyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::setup()] \""<<getName()<<"\": error during cudaMemcpy3D()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                res = cudaMemcpy2D( memory._devPtr, memory._pitch, data, getDimension(0) * getElementSize(), getDimension(0), getDimension(1), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::setup()] \""<<getName()<<"\": error during cudaMemcpy2D()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                } 
            }
            else
            {
                res = cudaMemcpy( memory._devPtr,  data, getByteSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::setup()] \""<<getName()<<"\": error during cudaMemcpy()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }

            // host must be synchronized
            // because device memory has been modified
            memory._syncOp = osgCompute::SYNC_HOST;
            memory._modifyCount = _image.valid()? _image->getModifiedCount() : UINT_MAX;
            memory._modifyCount = _array.valid()? _array->getModifiedCount() : UINT_MAX;

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
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::setup()] \""<<getName()<<"\": cannot receive valid data pointer."
                    << std::endl;

                return false;
            }

            res = cudaMemcpy( memory._hostPtr,  data, getByteSize(), cudaMemcpyHostToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::setup()] \""<<getName()<<"\": error during cudaMemcpy()."
                    << " " << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            // device must be synchronized
            // because host memory has been modified
            memory._syncOp = osgCompute::SYNC_DEVICE;
            memory._modifyCount = _image.valid()? _image->getModifiedCount() : UINT_MAX;
            memory._modifyCount = _array.valid()? _array->getModifiedCount() : UINT_MAX;

            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool Buffer::alloc( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        BufferObject* memoryPtr = dynamic_cast<BufferObject*>( object() );
        if( !memoryPtr )
            return false;
        BufferObject& memory = *memoryPtr;

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
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::alloc()] \""<<getName()<<"\": error during malloc()."
                    << std::endl;

                return false;
            }

            // clear memory
            memset( memory._hostPtr, 0x0, getByteSize() );

            if( memory._devPtr != NULL )
                memory._syncOp |= osgCompute::SYNC_HOST;

            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            if( memory._devPtr != NULL )
                return true;

            if( getNumDimensions() == 3 )
            {
                cudaPitchedPtr pitchPtr;
                cudaExtent extent;
                extent.width = getDimension(0) * getElementSize();
                extent.height = getDimension(1);
                extent.depth = getDimension(2);

                // allocate memory
                cudaError_t res = cudaMalloc3D( &pitchPtr, extent );
                if( cudaSuccess != res || NULL == pitchPtr.ptr )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::alloc()] \""<<getName()<<"\": error during mallocDevice3D()."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }

                memory._devPtr = pitchPtr.ptr;
                memory._pitch = pitchPtr.pitch;

                // clear memory
                cudaMemset3D( pitchPtr, 0x0, extent );
            }
            else if( getNumDimensions() == 2 )
            {
                cudaError_t res = cudaMallocPitch( &memory._devPtr, (size_t*)&memory._pitch, getDimension(0) * getElementSize(), getDimension(1) );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::alloc()] \""<<getName()<<"\": error during mallocDevice2D()."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }


                // clear memory
                cudaMemset2D( memory._devPtr, memory._pitch, 0x0, getDimension(0), getDimension(1) );
            }
            else
            {
                cudaError_t res = cudaMalloc( &memory._devPtr, getByteSize() );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::alloc()] \""<<getName()<<"\": error during mallocDevice()."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return false;
                }

                memory._pitch = getByteSize();
                // clear memory
                cudaMemset( memory._devPtr, 0x0, getByteSize() );
            }

            if( memory._pitch != (getDimension(0) * getElementSize()) )
            {
                int device = 0;
                cudaGetDevice( &device );
                int textureAlignment = 0;
                cudaDeviceProp devProp;
                cudaGetDeviceProperties( &devProp, device );

                osg::notify(osg::INFO)
                    << getName() << " [osgCuda::Buffer::alloc()] \""<<getName()
                    << "\": Memory requirement is not a multiple of texture alignment. This "
                    << "leads to a pitch which is not equal to the logical row size in bytes. "
                    << "Texture alignment requirement is \""
                    << devProp.textureAlignment << "\". "
                    << std::endl;
            }

            if( memory._hostPtr != NULL )
                memory._syncOp |= osgCompute::SYNC_DEVICE;

            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool Buffer::sync( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        BufferObject* memoryPtr = dynamic_cast<BufferObject*>( object() );
        if( !memoryPtr )
            return false;
        BufferObject& memory = *memoryPtr;

        /////////////////
        // SYNC MEMORY //
        /////////////////
        if( mapping & osgCompute::MAP_DEVICE )
        {
            if( !(memory._syncOp & osgCompute::SYNC_DEVICE) )
                return true;

            if( getNumDimensions() == 3 )
            {
                cudaMemcpy3DParms memCpyParams = {0};
                memCpyParams.dstPtr = make_cudaPitchedPtr(memory._devPtr,memory._pitch, getDimension(0), getDimension(1));
                memCpyParams.kind = cudaMemcpyHostToDevice;
                memCpyParams.srcPtr = make_cudaPitchedPtr(memory._hostPtr,getElementSize()*getDimension(0), getDimension(0), getDimension(1));

                cudaExtent arrayExtent = {0};
                arrayExtent.width = getElementSize()*getDimension(0);
                arrayExtent.height = getDimension(1);
                arrayExtent.depth = getDimension(2);

                memCpyParams.extent = arrayExtent;

                res = cudaMemcpy3D( &memCpyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << "[osgCuda::Buffer::sync()]: cudaMemcpy3D() to device failed."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                res = cudaMemcpy2D( memory._devPtr, memory._pitch, memory._hostPtr, getElementSize()*getDimension(0), 
                    getDimension(0), getDimension(1), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Buffer::sync()]: cudaMemcpy2D() to device failed."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                res = cudaMemcpy( memory._devPtr, memory._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::sync()] \""<<getName()<<"\": cudaMemcpy() to device failed."
                        << " " << cudaGetErrorString( res ) <<"."
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
                cudaMemcpy3DParms memCpyParams = {0};
                memCpyParams.dstPtr = make_cudaPitchedPtr(memory._hostPtr,getElementSize()*getDimension(0), getDimension(0), getDimension(1));
                memCpyParams.kind = cudaMemcpyDeviceToHost;
                memCpyParams.srcPtr = make_cudaPitchedPtr(memory._devPtr,memory._pitch, getDimension(0), getDimension(1));

                cudaExtent arrayExtent = {0};
                arrayExtent.width = getElementSize()*getDimension(0);
                arrayExtent.height = getDimension(1);
                arrayExtent.depth = getDimension(2);

                memCpyParams.extent = arrayExtent;

                res = cudaMemcpy3D( &memCpyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << "[osgCuda::Buffer::sync()]: cudaMemcpy3D() to host failed."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else if( getNumDimensions() == 2 )
            {
                res = cudaMemcpy2D( memory._hostPtr, getElementSize()*getDimension(0), memory._devPtr, memory._pitch, 
                    getDimension(0), getDimension(1), cudaMemcpyDeviceToHost );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::Buffer::sync()]: cudaMemcpy2D() to host failed."
                        << " " << cudaGetErrorString( res ) << "."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                res = cudaMemcpy( memory._hostPtr, memory._devPtr, getByteSize(), cudaMemcpyDeviceToHost );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::Buffer::sync()] \""<<getName()<<"\": cudaMemcpy() to host failed."
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
    void Buffer::setImage( osg::Image* image )
    {
        if( !osgCompute::Resource::isClear() && image != NULL )
        {
            if( image->getNumMipmapLevels() > 1 )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::setImage()] \""<<getName()<<"\": image \""
                    << image->getName() << "\" uses MipMaps which are currently"
                    << "not supported." 
                    << std::endl;

                return;
            }

            if( image->getTotalSizeInBytes() != getByteSize() )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::setImage()] \""<<getName()<<"\": size of image \""
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
    osg::Image* Buffer::getImage()
    {
        return _image.get();
    }

    //------------------------------------------------------------------------------
    const osg::Image* Buffer::getImage() const
    {
        return _image.get();
    }

    //------------------------------------------------------------------------------
    void Buffer::setArray( osg::Array* array )
    {
        if( !osgCompute::Resource::isClear() && array != NULL )
        {
            if( array->getTotalDataSize() != getByteSize() )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::Buffer::setArray()] \""<<getName()<<"\": size of array \""
                    << array->getName() << "\" is wrong."
                    << std::endl;

                return;
            }
        }

        _array = array;
        _image = NULL;
        resetModifiedCounts();
    }

    //------------------------------------------------------------------------------
    osg::Array* Buffer::getArray()
    {
        return _array.get();
    }

    //------------------------------------------------------------------------------
    const osg::Array* Buffer::getArray() const
    {
        return _array.get();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Buffer::clearLocal()
    {
        _array = NULL;
        _image = NULL;
    }

    //------------------------------------------------------------------------------
    unsigned int Buffer::computePitch() const
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
    osgCompute::MemoryObject* Buffer::createObject() const
    {
        return new BufferObject;
    }

    //------------------------------------------------------------------------------
    void Buffer::resetModifiedCounts() const
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        const BufferObject* memoryPtr = dynamic_cast<const BufferObject*>( object() );
        if( !memoryPtr )
            return;
        BufferObject& memory = const_cast<BufferObject&>(*memoryPtr);

        ///////////////////
        // RESET COUNTER //
        ///////////////////
        memory._modifyCount = UINT_MAX;
    }
}
