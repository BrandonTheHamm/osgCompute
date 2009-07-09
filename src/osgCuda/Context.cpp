#include <malloc.h>
#include <map>
#include <set>
#include <string.h>
#include <osg/Notify>
#include <osg/GL>
#include <osg/GraphicsThread>
#include <osg/GraphicsContext>
#include <osg/State>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include "osgCuda/Context"

namespace osgCuda
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    static OpenThreads::Mutex           s_sharedMutex;
    static std::set< int >              s_sharedDevices;

    static bool setupSharedDevice( int device )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_sharedMutex);
        std::set< int >::iterator itr = s_sharedDevices.find( device );
        if( itr != s_sharedDevices.end() )
            return true;

        cudaError res = cudaGLSetGLDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "CUDA::Context::init(): cannot share device with render context."
                << std::endl;

            return false;
        }

        s_sharedDevices.insert( device );
        return true;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    // DECLARATIONS //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    struct RegMem;
    struct AllocMem;

    typedef std::map<void*,AllocMem>                          AllocMap;
    typedef std::map<void*,AllocMem>::iterator                AllocMapItr;
    typedef std::map<void*,AllocMem>::const_iterator          AllocMapCnstItr;

    typedef std::map<GLuint,RegMem>                           RegistrationMap;
    typedef std::map<GLuint,RegMem>::iterator                 RegistrationMapItr;
    typedef std::map<GLuint,RegMem>::const_iterator           RegistrationMapCnstItr;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PRIVATE OBJECTS ///////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    enum MallocType
    {
        MALLOC_TYPE_UNKNOWN          = 0,
        MALLOC_TYPE_HOST             = 1,
        MALLOC_TYPE_DEVICE           = 2,
        MALLOC_TYPE_DEVICE_HOST      = 3,
        MALLOC_TYPE_DEVICE_2D        = 4,
        MALLOC_TYPE_DEVICE_3D        = 5,
        MALLOC_TYPE_DEVICE_ARRAY     = 6,
        MALLOC_TYPE_DEVICE_ARRAY_2D  = 7,
        MALLOC_TYPE_DEVICE_ARRAY_3D  = 8,
    };

    struct AllocMem
    {
        union
        {
            void*                 _buffer;
            cudaArray*            _array;
        };

        unsigned int          _bytes;
        MallocType            _type;

        AllocMem() { memset(this,0x0,sizeof(AllocMem)); }
    };

    struct RegMem
    {
        GLuint                _bo;
        unsigned int          _bytes;
        bool                  _allocated;

        RegMem() { memset(this,0x0,sizeof(RegMem)); }
    };

    struct ContextMemory
    {
        OpenThreads::Mutex           _mutex;
        AllocMap                     _allocMem;
        RegistrationMap              _regMem;
        AllocMap                     _lazyFreeMem;
        RegistrationMap              _lazyUnregMem;
    };



    /**
    */
    class ReleaseOperation: public osg::Operation
    {
    public:
        ReleaseOperation(osg::State& state, Context& context)
            : osg::Operation("CudaReleaseOperation",true), _state(&state), _context(&context)
        {
        }

        //------------------------------------------------------------------------------
        virtual void operator () (osg::Object*)
        {
            // just free the unused memory
            _context->clearMemory();

            return;
        }

        //------------------------------------------------------------------------------
        // any CUDA resources created through the runtime in one host thread cannot be
        // used by the runtime from another host thread.
        virtual void release()
        {
            if( _context.valid() )
            {
                _context->clear();
                _context = NULL;
            }

            if( _state.valid() )
                _state = NULL;
        }

    protected:

        virtual ~ReleaseOperation() { release(); }

        osg::ref_ptr<Context>       _context;
        osg::ref_ptr<osg::State>    _state;
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Context::Context()
        : osgCompute::Context(),
          _asgThread(NULL),
          _ctxmem(NULL)
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool Context::init()
    {
        if( !isClear() )
            return false;

        //////////////////
        // PROOF DEVICE //
        //////////////////
        // enumerate devices
        int deviceCount = 0;
        cudaError res = cudaGetDeviceCount( &deviceCount );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  << "CUDA::Context::init(): something goes wrong on cudaGetDeviceCount(). Returned code is "<<res<<"."
                << std::endl;
            clear();
            return false;
        }

        if( _device > deviceCount - 1 )
        {
            osg::notify(osg::FATAL)  << "CUDA::Context::init(): device \""<<_device<<"\" does not exist."
                << std::endl;

            clear();
            return false;
        }

        cudaDeviceProp deviceProp;
        res = cudaGetDeviceProperties( &deviceProp, _device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  << "CUDA::Context::init(): no device found which supports CUDA."
                << std::endl;

            clear();
            return false;
        }

        if( deviceProp.major < 1 )
        {
            osg::notify(osg::FATAL)  << "CUDA::Context::init(): device does not support CUDA.\n"
                << std::endl;

            clear();
            return false;
        }

        _deviceProperties = deviceProp;

        ////////////////
        // CTX MEMORY //
        ////////////////
        _ctxmem = new ContextMemory;
        if( !_ctxmem )
        {
            clear();
            return false;
        }

        // In case the context is used in combination with a osg::State object
        // and within a multi threaded environment add a release operation to
        // the current graphics thread.
        if( getState() != NULL )
        {
            //////////////////
            // SHARE DEVICE //
            //////////////////
            if( !setupSharedDevice(_device) )
            {
                osg::notify(osg::FATAL)  
                    << "CUDA::Context::init(): cannot share device with render context."
                    << std::endl;

                clear();
                return false;
            }

            ///////////////////
            // ADD OPERATION //
            ///////////////////
            osg::GraphicsThread* gct = NULL;

            osg::GraphicsContext* gc = getState()->getGraphicsContext();
            if( gc )
                gct = gc->getGraphicsThread();

            if( gct )
            {
                _releaseOp = new ReleaseOperation( *getState(), *this );
                gct->add( _releaseOp );

                // graphics thread is the only thread that is allowed to call apply
                _asgThread = gct;
            }
        }

        return osgCompute::Context::init();
    }

    //------------------------------------------------------------------------------
    void Context::apply()
    {
        // first apply determines the related thread
        if( _asgThread == NULL )
            _asgThread = OpenThreads::Thread::CurrentThread();

        // return if different thread calls apply()
        if( _asgThread != OpenThreads::Thread::CurrentThread() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::apply(): context cannot applied to different threads."
                << std::endl;

            return;
        }

        osgCompute::Context::apply();
    }

    //------------------------------------------------------------------------------
    void* Context::mallocHostMemory( unsigned int byteSize ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocHostMemory(): context is dirty."
                << std::endl;

            return NULL;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocHostMemory(): cannot allocate resources from different threads.\n"
                << std::endl;

            return NULL;
        }

        // allocate memory
        AllocMem allocMem;
        allocMem._buffer = malloc( byteSize );
        if( allocMem._buffer == NULL )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocHostMemory() for Context \""<< getId()<<"\": something goes wrong within malloc()."
                << std::endl;

            return NULL;
        }

        // setup params
        allocMem._bytes = byteSize;
        allocMem._type = MALLOC_TYPE_HOST;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( allocMem._buffer, allocMem ) );
        }
        return allocMem._buffer;
    }

    //------------------------------------------------------------------------------
    void* Context::mallocDeviceMemory( unsigned int byteSize ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDeviceMemory(): context is dirty."
                << std::endl;

            return NULL;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDeviceMemory(): cannot allocate resources from different threads.\n"
                << std::endl;

            return NULL;
        }

        // allocate memory
        AllocMem allocMem;
        cudaError_t res = cudaMalloc( &allocMem._buffer, byteSize );
        if( res != cudaSuccess )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDeviceMemory() for Context \""<< getId()<<"\": something goes wrong within cudaMalloc()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        allocMem._bytes = byteSize;
        allocMem._type = MALLOC_TYPE_DEVICE;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( allocMem._buffer, allocMem ) );
        }
        return allocMem._buffer;
    }

    //------------------------------------------------------------------------------
    void* Context::mallocDeviceHostMemory( unsigned int byteSize ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDeviceHostMemory(): context is dirty."
                << std::endl;

            return NULL;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDeviceHostMemory(): cannot allocate resources from different threads.\n"
                << std::endl;

            return NULL;
        }

        // allocate memory
        AllocMem allocMem;
        cudaError_t res = cudaMallocHost( &allocMem._buffer, byteSize );
        if( res != cudaSuccess )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDeviceHostMemory() for Context \""<< getId()<<"\": something goes wrong within cudaMallocHost()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        allocMem._bytes = byteSize;
        allocMem._type = MALLOC_TYPE_DEVICE_HOST;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( allocMem._buffer, allocMem ) );
        }
        return allocMem._buffer;
    }

    //------------------------------------------------------------------------------
    void* Context::mallocDevice2DMemory( unsigned int widthPitch, unsigned int height ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDevice2DMemory(): context is dirty."
                << std::endl;

            return NULL;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDevice2DMemory(): cannot allocate resources from different threads.\n"
                << std::endl;

            return NULL;
        }

        size_t pitch = 0;
        AllocMem allocMem;
        cudaError_t res = cudaMallocPitch( &allocMem._buffer, &pitch, widthPitch, height );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDevice2DMemory() for Context \""<< getId()<<"\": something goes wrong within cudaMallocPitch()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        allocMem._bytes = widthPitch*height;
        allocMem._type = MALLOC_TYPE_DEVICE_2D;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( allocMem._buffer, allocMem ) );
        }
        return allocMem._buffer;
    }

    //------------------------------------------------------------------------------
    void* Context::mallocDevice3DMemory(  unsigned int widthPitch, unsigned int height, unsigned int depth ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDevice3DMemory(): context is dirty."
                << std::endl;

            return NULL;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::mallocDevice3DMemory(): cannot allocate resources for different threads.\n"
                << std::endl;

            return NULL;
        }

        cudaPitchedPtr pitchPtr;
        cudaExtent extent;
        extent.width = widthPitch;
        extent.height = height;
        extent.depth = depth;

        // allocate memory
        AllocMem allocMem;
        cudaError_t res = cudaMalloc3D( &pitchPtr, extent );
        if( cudaSuccess != res || NULL == pitchPtr.ptr )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDevice3DMemory() for Context \""<< getId()<<"\": something goes wrong within cudaMalloc3D()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        allocMem._bytes = widthPitch * height * depth;
        allocMem._type = MALLOC_TYPE_DEVICE_3D;
        allocMem._buffer = pitchPtr.ptr;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( allocMem._buffer, allocMem ) );
        }
        return allocMem._buffer;
    }

    //------------------------------------------------------------------------------
    cudaArray* Context::mallocDeviceArray( unsigned int width, const cudaChannelFormatDesc& desc ) const
    {
        // allocate memory
        AllocMem allocMem;
        cudaError_t res = cudaMallocArray( &allocMem._array, &desc, width, 1 );
        if( cudaSuccess != res || NULL ==  allocMem._array )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDeviceArray() for Context \""<< getId()<<"\": Something goes wrong within cudaMallocArray()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        unsigned int elementBitSize = static_cast<unsigned int>(desc.x + desc.y + desc.z + desc.w);
        allocMem._bytes = static_cast<unsigned int>(elementBitSize/8);
        if( elementBitSize % 8 != 0 )
            allocMem._bytes++;
        allocMem._bytes *= width;
        allocMem._type = MALLOC_TYPE_DEVICE_ARRAY;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( reinterpret_cast<void*>(allocMem._array), allocMem ) );
        }
        return allocMem._array;
    }

    //------------------------------------------------------------------------------
    cudaArray* Context::mallocDevice2DArray( unsigned int width, unsigned int height, const cudaChannelFormatDesc& desc ) const
    {
        // allocate memory
        AllocMem allocMem;
        cudaError_t res = cudaMallocArray( &allocMem._array, &desc, width, height );
        if( cudaSuccess != res || NULL ==  allocMem._array )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDevice2DArray() for Context \""<< getId()<<"\": Something goes wrong within cudaMallocArray()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        unsigned int elementBitSize = static_cast<unsigned int>(desc.x + desc.y + desc.z + desc.w);
        allocMem._bytes = static_cast<unsigned int>(elementBitSize/8);
        if( elementBitSize % 8 != 0 )
            allocMem._bytes++;
        allocMem._bytes *= width;
        allocMem._bytes *= height;
        allocMem._type = MALLOC_TYPE_DEVICE_ARRAY_2D;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( reinterpret_cast<void*>(allocMem._array), allocMem ) );
        }
        return allocMem._array;
    }

    //------------------------------------------------------------------------------
    cudaArray* Context::mallocDevice3DArray( unsigned int width, unsigned int height, unsigned int depth, const cudaChannelFormatDesc& desc ) const
    {
        cudaExtent extent;
        extent.width = width;
        extent.height = height;
        extent.depth = depth;

        // allocate memory
        AllocMem allocMem;
        cudaError_t res = cudaMalloc3DArray( &allocMem._array, &desc, extent );
        if( cudaSuccess != res || NULL ==  allocMem._array )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocDevice3DArray() for Context \""<< getId()<<"\": Something goes wrong within cudaMalloc3DArray()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return NULL;
        }

        // setup params
        unsigned int elementBitSize = static_cast<unsigned int>(desc.x + desc.y + desc.z + desc.w);
        allocMem._bytes = static_cast<unsigned int>(elementBitSize/8);
        if( elementBitSize % 8 != 0 )
            allocMem._bytes++;
        allocMem._bytes *= width;
        allocMem._bytes *= height;
        allocMem._bytes *= depth;
        allocMem._type = MALLOC_TYPE_DEVICE_ARRAY_3D;

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_allocMem.insert( std::make_pair<void*, AllocMem>( reinterpret_cast<void*>(allocMem._array), allocMem ) );
        }
        return allocMem._array;
    }

    //------------------------------------------------------------------------------
    GLuint Context::mallocBufferObject( unsigned int byteSize ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::registerBufferObject(): context is dirty."
                << std::endl;

            return UINT_MAX;
        }

        if( !getState() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::registerBufferObject(): No valid osg::State found."
                << std::endl;

            return UINT_MAX;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::registerBufferObject(): cannot register resources for different threads.\n"
                << std::endl;

            return UINT_MAX;
        }


        osg::BufferObject::Extensions* bufferExt = osg::BufferObject::getExtensions( getState()->getContextID(),true );
        ////////////////
        // CREATE PBO //
        ////////////////
        RegMem regMem;
        regMem._bo = 0;
        regMem._bytes = byteSize;
        regMem._allocated = true;

        bufferExt->glGenBuffers( 1, &regMem._bo );
        GLenum errorNo = glGetError();
        if( 0 == regMem._bo  || errorNo )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocBufferObject() for Context \""
                << getId()<<"\": cannot generate BufferObject (glGenBuffers())."
                << std::endl;

            return UINT_MAX;
        }

        ////////////////////
        // INITIALIZE PBO //
        ////////////////////
        // Allocate temporary memory
        void *tmpData = malloc(byteSize);
        memset( tmpData, 0x0, byteSize );

        // Initialize PixelBufferObject
        bufferExt->glBindBuffer( GL_ARRAY_BUFFER_ARB, regMem._bo );
        errorNo = glGetError();
        if (errorNo != GL_NO_ERROR)
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocBufferObject() for Context \""
                << getId()<<"\": cannot bind BufferObject (glBindBuffer())."
                << std::endl;

            return UINT_MAX;
        }

        bufferExt->glBufferData( GL_ARRAY_BUFFER_ARB, byteSize, tmpData, GL_DYNAMIC_DRAW );
        errorNo = glGetError();
        if (errorNo != GL_NO_ERROR)
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::mallocBufferObject() for Context \""
                << getId()<<"\": cannot initialize BufferObject (glBufferData())."
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
        cudaError res = cudaGLRegisterBufferObject( regMem._bo );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::registerBufferObject() for Context \""<< getId()<<"\": something goes wrong within cudaGLRegisterBufferObject()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return UINT_MAX;
        }

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_regMem.insert( std::make_pair<GLuint, RegMem>( regMem._bo, regMem ) );
        }

        return regMem._bo;
    }

    //------------------------------------------------------------------------------
    bool Context::registerBufferObject( GLuint bo, unsigned int byteSize ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::registerBufferObject(): context is dirty."
                << std::endl;

            return false;
        }

        if( !getState() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::registerBufferObject(): no valid osg::State found."
                << std::endl;

            return false;
        }

        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::registerBufferObject(): cannot register resources for different threads.\n"
                << std::endl;

            return false;
        }

        RegMem regMem;
        regMem._bo = bo;
        regMem._bytes = byteSize;
        regMem._allocated = false;

        cudaError res = cudaGLRegisterBufferObject( regMem._bo );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::registerBufferObject() for Context \""<< getId()<<"\": something goes wrong within cudaGLRegisterBufferObject()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return false;
        }

        // insert and return
        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
            _ctxmem->_regMem.insert( std::make_pair<GLuint, RegMem>( bo, regMem ) );
        }

        return true;
    }

    //------------------------------------------------------------------------------
    void Context::freeMemory( void* buffer ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::freeMemory(\"BUFFER\"): context is dirty."
                << std::endl;

            return;
        }

        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);

        AllocMapItr itr = _ctxmem->_allocMem.find( buffer );
        if( itr == _ctxmem->_allocMem.end() )
            return;

        AllocMem& allocMem = (*itr).second;
        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            // free memory later
            _ctxmem->_lazyFreeMem.insert( std::make_pair<void*,AllocMem>(buffer, allocMem) );
            _ctxmem->_allocMem.erase( itr );
            return;
        }

        // free memory now
        cudaError_t res = cudaErrorUnknown;
        switch( allocMem._type )
        {
        case MALLOC_TYPE_DEVICE_3D:
        case MALLOC_TYPE_DEVICE_2D:
        case MALLOC_TYPE_DEVICE:
            res = cudaFree( allocMem._buffer );
            break;
        case MALLOC_TYPE_DEVICE_HOST:
            res = cudaFreeHost( allocMem._buffer );
            break;
        case MALLOC_TYPE_HOST:
            free( allocMem._buffer );
            res = cudaSuccess;
            break;
        default:
            break;
        }

        if( res != cudaSuccess )
        {
             osg::notify(osg::FATAL)
                << "CUDA::Context::freeMemory(\"BUFFER\") for Context \""<< getId()<<"\": something goes wrong during deallocation."
                << "Returned code is " << std::hex << res << "."
                << std::endl;
        }

        _ctxmem->_allocMem.erase( itr );
    }

    //------------------------------------------------------------------------------
    void Context::freeMemory( cudaArray* array ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::freeMemory(\"ARRAY\"): context is dirty."
                << std::endl;

            return;
        }

        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);
        AllocMapItr itr = _ctxmem->_allocMem.find( reinterpret_cast<void*>(array) );
        if( itr == _ctxmem->_allocMem.end() )
            return;

        AllocMem& allocMem = (*itr).second;
        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            // free memory later
            _ctxmem->_lazyFreeMem.insert( std::make_pair<void*,AllocMem>( reinterpret_cast<void*>(array), allocMem ) );
            _ctxmem->_allocMem.erase( itr );
            return;
        }

        // free array now
        cudaError_t res = cudaErrorUnknown;
        switch( allocMem._type )
        {
        case MALLOC_TYPE_DEVICE_ARRAY_3D:
        case MALLOC_TYPE_DEVICE_ARRAY_2D:
        case MALLOC_TYPE_DEVICE_ARRAY:
            res = cudaFreeArray( allocMem._array );
            break;
        default:
            break;
        }

        if( res != cudaSuccess )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::freeMemory(\"ARRAY\") for Context \""<< getId()<<"\": something goes wrong during deallocation."
                << "Returned code is " << std::hex << res << "."
                << std::endl;
        }

        _ctxmem->_allocMem.erase( itr );
    }

    //------------------------------------------------------------------------------
    void Context::freeBufferObject( GLuint bo ) const
    {
        if( isClear() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::unregisterBufferObject(): context is dirty."
                << std::endl;

            return;
        }

        if( !getState() )
        {
            osg::notify(osg::WARN)
                << "CUDA::Context::unregisterBufferObject(): no valid osg::State found."
                << std::endl;

            return;
        }

        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_ctxmem->_mutex);

        RegistrationMapItr itr = _ctxmem->_regMem.find( bo );
        if( itr == _ctxmem->_regMem.end() )
            return;

        RegMem& regMem = (*itr).second;
        OpenThreads::Thread* curThread = OpenThreads::Thread::CurrentThread();
        if( curThread != _asgThread )
        {
            // free memory later
            _ctxmem->_lazyUnregMem.insert( std::make_pair<GLuint,RegMem>(bo, regMem) );
            _ctxmem->_regMem.erase( itr );
            return;
        }

        cudaError_t res = cudaGLUnregisterBufferObject( bo );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)
                << "CUDA::Context::unregisterBufferObject for Context \""
                << getId()<<"\": something goes wrong within cudaGLUnregisterBufferObject()."
                << "Returned code is " << std::hex << res << "."
                << std::endl;

            return;
        }

        // free buffer object
        if( regMem._allocated )
        {
            // ... delete buffer object
            osg::BufferObject::Extensions* ext = osg::BufferObject::getExtensions( getState()->getContextID(),true);
            ext->glBindBuffer( GL_ARRAY_BUFFER_ARB, regMem._bo );
            ext->glDeleteBuffers( 1, &regMem._bo );
        }

        _ctxmem->_regMem.erase( itr );
    }

    //------------------------------------------------------------------------------
    void Context::clear()
    {
        // do not switch the following order!!
        osgCompute::Context::clear();
        clearLocal();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Context::clearLocal()
    {
        clearMemory();

        if( _ctxmem )
        {
            if( !_ctxmem->_allocMem.empty() )
            {
                osg::notify(osg::FATAL)
                    << "CUDA::Context::clearLocal(): resources of size \"";

                unsigned int byteSize = 0;
                for( AllocMapCnstItr itr = _ctxmem->_allocMem.begin();
                    itr != _ctxmem->_allocMem.end(); ++itr )
                {
                    byteSize += (*itr).second._bytes;
                }

                osg::notify(osg::FATAL)
                    << byteSize
                    << "\" are still allocated."
                    << std::endl;
            }

            if( !_ctxmem->_lazyFreeMem.empty() )
            {
                osg::notify(osg::FATAL)
                    << "CUDA::Context::clearLocal(): resources of size \"";

                unsigned int byteSize = 0;
                for( AllocMapCnstItr itr = _ctxmem->_allocMem.begin();
                    itr != _ctxmem->_allocMem.end(); ++itr )
                {
                    byteSize += (*itr).second._bytes;
                }

                osg::notify(osg::FATAL)
                    << byteSize
                    << "\" cannot be freed at all."
                    << std::endl;
            }

            if( !_ctxmem->_lazyUnregMem.empty() )
            {
                osg::notify(osg::FATAL)
                    << "CUDA::Context::clearLocal(): resources of size \"";

                unsigned int byteSize = 0;
                for( RegistrationMapCnstItr itr = _ctxmem->_regMem.begin();
                    itr != _ctxmem->_regMem.end(); ++itr )
                {
                    byteSize += (*itr).second._bytes;
                }

                osg::notify(osg::FATAL)
                    << byteSize
                    << "\" cannot be unregistered at all."
                    << std::endl;
            }

            _ctxmem->_allocMem.clear();
            _ctxmem->_lazyFreeMem.clear();

            delete _ctxmem;
            _ctxmem = NULL;
        }

        memset( &_deviceProperties, 0x0, sizeof(cudaDeviceProp) );
        _asgThread = NULL;
        _releaseOp = NULL;
    }

    //------------------------------------------------------------------------------
    void Context::clearMemory()
    {
        if( _asgThread == NULL ||
            _asgThread != OpenThreads::Thread::CurrentThread() )
            return;

        if( !_ctxmem )
            return;

        cudaError_t res = cudaSuccess;

        //free all memory
        while( !_ctxmem->_lazyFreeMem.empty() )
        {
            AllocMapItr itr = _ctxmem->_lazyFreeMem.begin();
            AllocMem& allocMem = (*itr).second;

            // free buffer
            if( allocMem._buffer != NULL )
            {
                switch( allocMem._type )
                {
                case MALLOC_TYPE_HOST:
                    {
                        free( allocMem._buffer );
                        allocMem._buffer = NULL;
                    }
                    break;
                case MALLOC_TYPE_DEVICE:
                case MALLOC_TYPE_DEVICE_2D:
                case MALLOC_TYPE_DEVICE_3D:
                    {
                        res = cudaFree( allocMem._buffer );
                        allocMem._buffer = NULL;
                    }
                    break;
                case MALLOC_TYPE_DEVICE_HOST:
                    {
                        res = cudaFreeHost( allocMem._buffer );
                        allocMem._buffer = NULL;
                    }
                    break;
                case MALLOC_TYPE_DEVICE_ARRAY_3D:
                case MALLOC_TYPE_DEVICE_ARRAY_2D:
                case MALLOC_TYPE_DEVICE_ARRAY:
                    {
                        res = cudaFreeArray(  allocMem._array );
                        allocMem._array = NULL;
                    }
                default: break;
                }
            }

            if( res != cudaSuccess )
            {
                osg::notify(osg::WARN)
                    << "CUDA::Context::clearMemory(): freeing \""<<allocMem._bytes<<"\" bytes of memory failed."
                    << " Returned code is \""<<std::hex << res << "\"."
                    << std::endl;
            }

            _ctxmem->_lazyFreeMem.erase( itr );
        }

        // unregister memory
        if( getState() )
        {
            while( !_ctxmem->_lazyUnregMem.empty() )
            {
                RegistrationMapItr itr = _ctxmem->_lazyUnregMem.begin();
                RegMem& regMem = (*itr).second;

                res = cudaGLUnregisterBufferObject( regMem._bo );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::WARN)
                        << "CUDA::Context::clearMemory(): unregistering \""<<regMem._bytes<<"\" bytes of memory failed."
                        << " Returned code is \""<<std::hex << res << "\"."
                        << std::endl;
                }

                if( regMem._allocated )
                {
                    // ... delete buffer object
                    osg::BufferObject::Extensions* ext = osg::BufferObject::getExtensions( getState()->getContextID(),true);
                    ext->glBindBuffer( GL_ARRAY_BUFFER_ARB, regMem._bo );
                    ext->glDeleteBuffers( 1, &regMem._bo );
                }

                _ctxmem->_lazyUnregMem.erase( itr );
            }
        }
    }
}