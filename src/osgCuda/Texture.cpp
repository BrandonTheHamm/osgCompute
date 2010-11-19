#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
#include <osg/GL>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osgCuda/Texture>

namespace osgCuda
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    TextureObject::TextureObject()
        : osgCompute::MemoryObject(),
          _hostPtr(NULL),
          _devPtr(NULL),
          _graphicsArray(NULL),
          _graphicsResource(NULL),
          _lastModifiedCount(UINT_MAX),
		  _lastModifiedAddress(UINT_MAX)
    {
    }

    //------------------------------------------------------------------------------
    TextureObject::~TextureObject()
    {
        if( _devPtr != NULL )
        {
            cudaError res = cudaFree( _devPtr );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                <<"[TextureObject::~TextureObject()]: error during cudaFree()."
                << cudaGetErrorString(res) << std::endl;
            }
        }

        if( _graphicsArray != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &_graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::WARN)
                    << "[TextureObject::~TextureObject()]: error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
        }


        if( _graphicsResource != NULL )
        {
            cudaError res = cudaGraphicsUnregisterResource( _graphicsResource );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                <<"[TextureObject::~TextureObject()]: error during cudaGraphicsUnregisterResource()."
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
    TextureBuffer::TextureBuffer()
        : osgCompute::InteropMemory()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    TextureBuffer::~TextureBuffer()
    {
        clearLocal();

        // free this proxy object
        getInteropObject()->freeProxy();
        // decrease the reference count of the geometry reference
        _texref = NULL;
    }

    //------------------------------------------------------------------------------
    void TextureBuffer::clear()
    {
        osgCompute::InteropMemory::clear();
        clearLocal();
    }

	//------------------------------------------------------------------------------
	osgCompute::InteropObject* TextureBuffer::getInteropObject()
	{ 
		return dynamic_cast<osgCompute::InteropObject*>( _texref.get() );
	}

	//------------------------------------------------------------------------------
	const osgCompute::InteropObject* TextureBuffer::getInteropObject() const
	{ 
		return dynamic_cast<const osgCompute::InteropObject*>( _texref.get() );
	}

    //------------------------------------------------------------------------------
    bool TextureBuffer::init()
    {
        if( !isClear() )
            return true;

        if( !_texref.valid() )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::TextureBuffer::init()]: not connected to an texture object."
                << std::endl;

            clear();
            return false;
        }

        if( !initElementSize() )
        {
            clear();
            return false;
        }

        if( !initDimension() )
        {
            clear();
            return false;
        }

        return osgCompute::InteropMemory::init();
    }

    //------------------------------------------------------------------------------
    bool TextureBuffer::initDimension()
    {
        unsigned int dim[3];
        if( _texref->getImage(0) )
        {
            dim[0] = _texref->getImage(0)->s();
            dim[1] = _texref->getImage(0)->t();
            dim[2] = _texref->getImage(0)->r();
        }
        else
        {
            dim[0] = _texref->getTextureWidth();
            dim[1] = _texref->getTextureHeight();
            dim[2] = _texref->getTextureDepth();
        }

        if( dim[0] == 0 )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::TextureBuffer::initDimension()]: no dimensions defined for texture! set the texture dimensions first."
                << std::endl;

            return false;
        }

        unsigned int d = 0;
        while( dim[d] > 1 && d < 3 )
        {
            setDimension( d, dim[d] );
            ++d;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool TextureBuffer::initElementSize()
    {
        unsigned int elementSize = 0;
        unsigned int elementBitSize;
        if( _texref->getImage(0) )
        {
            elementBitSize = osg::Image::computePixelSizeInBits(
                _texref->getImage(0)->getPixelFormat(),
                _texref->getImage(0)->getDataType() );

        }
        else
        {
            GLenum format = osg::Image::computePixelFormat( _texref->getInternalFormat() );
            GLenum type = osg::Image::computeFormatDataType( _texref->getInternalFormat() );

            elementBitSize = osg::Image::computePixelSizeInBits( format, type );
        }

        elementSize = ((elementBitSize % 8) == 0)? elementBitSize/8 : elementBitSize/8 +1;
        if( elementSize == 0 )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::TextureBuffer::initElementSize()]: cannot determine element size."
                << std::endl;

            return false;
        }

        setElementSize( elementSize );
        return true;
    }

    //------------------------------------------------------------------------------
    void* TextureBuffer::map( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint/* = 0*/ )
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
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object() );
        if( !memoryPtr )
            return NULL;
        TextureObject& memory = *memoryPtr;

        if( _usage & osgCompute::GL_TARGET &&
            !(memory._syncOp & osgCompute::SYNC_ARRAY))
        {
            if( memory._graphicsResource == NULL )
            {
                // Initialize texture resource if it is a render target
                osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
                if( !tex )
                {
                    osg::State* state;
                    osg::GraphicsContext::GraphicsContexts _ctxs = osg::GraphicsContext::getAllRegisteredGraphicsContexts();
                    for( osg::GraphicsContext::GraphicsContexts::iterator itr = _ctxs.begin(); itr != _ctxs.end(); ++itr )
                    {
                        if( (*itr)->getState() && ((*itr)->getState()->getContextID() == osgCompute::Resource::getCurrentIdx()) )
                        {
                            state = (*itr)->getState();
                            break;
                        }
                    }

                    if( NULL == state )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to find valid state."
                            << std::endl;

                        return false;
                    }

                    _texref->compileGLObjects( *state );
                    tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
                }

                // Register vertex buffer object for Cuda
                cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to register image object (cudaGraphicsGLRegisterImage()). Not all GL formats are supported."
                        << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }


            // Host memory and shadow-copy should be synchronized in next call
            // We set this flag in general as we do not now if really has been
            // rendered to the texture
            memory._syncOp |= osgCompute::SYNC_DEVICE;
            memory._syncOp |= osgCompute::SYNC_HOST;
        }

        //////////////
        // MAP DATA //
        //////////////
        void* ptr = NULL;
        memory._mapping = mapping;
        bool firstLoad = false;

        // Check if image has changed.
		// Modified count is different if the data has changed and
		// modified address is different if the image object has changed
        bool needsSetup = false;
        if( _texref->getImage(0) != NULL && 
			( memory._lastModifiedCount != _texref->getImage(0)->getModifiedCount() || 
			  memory._lastModifiedAddress != (unsigned int) _texref->getImage(0) )
		   )
            needsSetup = true;

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

            //////////////////
            // SETUP STREAM //
            //////////////////
            if( needsSetup )
                if( !setup( mapping ) )
                    return NULL;

            /////////////////
            // SYNC STREAM //
            /////////////////
            if( memory._syncOp & osgCompute::SYNC_HOST )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._hostPtr;
        }
        else if( (mapping & osgCompute::MAP_DEVICE_ARRAY) == osgCompute::MAP_DEVICE_ARRAY )
        {
            //////////////////////
            // MAP ARRAY-MEMORY //
            //////////////////////
            // Create dynamic texture device memory
            // for each type of mapping
            if( NULL == memory._graphicsResource )
            {
                if( !alloc( mapping ) )
                    return NULL;

                firstLoad = true;
            }

            /////////////
            // MAP VBO //
            /////////////
            if( NULL == memory._graphicsArray )
            {
                cudaError res = cudaGraphicsMapResources(1, &memory._graphicsResource);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::TextureBuffer::map()]: error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }

                res = cudaGraphicsSubResourceGetMappedArray( &memory._graphicsArray, memory._graphicsResource, 0, 0);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << getName() << " [osgCuda::TextureBuffer::map()]: error during cudaGraphicsResourceGetMappedPointer(). "
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
            if( memory._syncOp & osgCompute::SYNC_ARRAY )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._graphicsArray;
        }
        else if( (mapping & osgCompute::MAP_DEVICE) )
        {
            ////////////////////////////
            // ALLOCATE DEVICE-MEMORY //
            ////////////////////////////
            if( NULL == memory._devPtr )
            {
                if( !alloc( mapping ) )
                    return NULL;

                firstLoad = true;
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
            if( memory._syncOp & osgCompute::SYNC_DEVICE )
                if( !sync( mapping ) )
                    return NULL;

            ptr = memory._devPtr;
        }
        else
        {
            osg::notify(osg::WARN)
                << getName() << " [osgCuda::TextureBuffer::map()]: Wrong mapping type specified. Use one of the following types: "
                << "HOST_SOURCE, HOST_TARGET, HOST, DEVICE_SOURCE, DEVICE_TARGET, DEVICE, DEVICE_ARRAY."
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
        {
            memory._syncOp |= osgCompute::SYNC_ARRAY;
            memory._syncOp |= osgCompute::SYNC_HOST;
        }
        else if( (mapping & osgCompute::MAP_HOST_TARGET) == osgCompute::MAP_HOST_TARGET )
        {
            memory._syncOp |= osgCompute::SYNC_ARRAY;
            memory._syncOp |= osgCompute::SYNC_DEVICE;
        }

        return &static_cast<char*>(ptr)[offset];
    }

    //------------------------------------------------------------------------------
    void TextureBuffer::unmap( unsigned int )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object() );
        if( !memoryPtr )
            return;
        TextureObject& memory = *memoryPtr;

        //////////////////
        // UNMAP MEMORY //
        //////////////////
        // Copy current memory to texture memory
        if( memory._syncOp & osgCompute::SYNC_ARRAY )
        {
            if( NULL == map( osgCompute::MAP_DEVICE_ARRAY, 0 ) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::unmap()]: error during device memory synchronization (map())."
                    << std::endl;

                return;
            }
        }

        if( memory._mapping == osgCompute::UNMAP && 
            _texref->getImage(0) != NULL &&
            _texref->getImage(0)->getModifiedCount() != memory._lastModifiedCount )
        {
            // Array is initialized during rendering. Sync others.
            memory._syncOp = osgCompute::SYNC_DEVICE | osgCompute::SYNC_HOST;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
        }

        // Change current context to render context
        if( memory._graphicsArray != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::WARN)
                    << getName() << " [osgCuda::TextureBuffer::unmap()]: error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            memory._graphicsArray = NULL;
        }

        memory._mapping = osgCompute::UNMAP;
    }

    //------------------------------------------------------------------------------
    bool TextureBuffer::reset( unsigned int  )
    {
        if( osgCompute::Resource::isClear() )
            if( !init() )
                return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object() );
        if( !memoryPtr )
            return NULL;
        TextureObject& memory = *memoryPtr;

        //////////////////
        // RESET MEMORY //
        //////////////////
        // Reset image data during the next mapping
        memory._lastModifiedCount = UINT_MAX;
        memory._syncOp = osgCompute::NO_SYNC;

        // Reset host memory
        if( memory._hostPtr != NULL && _texref->getImage(0) == NULL )
        {
            if( !memset( memory._hostPtr, 0x0, getByteSize() ) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::reset()]: error during memset() for host memory."
                    << std::endl;

                return false;
            }
        }

        // clear shadow-copy memory
        if( memory._devPtr != NULL && _texref->getImage(0) == NULL )
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
    bool TextureBuffer::isMappingAllowed( unsigned int mapping, unsigned int ) const
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
        case osgCompute::MAP_DEVICE_ARRAY:
            return true;
        default:
            return false;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void TextureBuffer::clearLocal()
    {
        // Do not call _texref = NULL;
        // Do not change _usage;
    }

    //------------------------------------------------------------------------------
    unsigned int TextureBuffer::computePitch() const
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
    bool TextureBuffer::setup( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object() );
        if( !memoryPtr )
            return NULL;
        TextureObject& memory = *memoryPtr;


        //////////////////
        // SETUP MEMORY //
        //////////////////
        if( (mapping & osgCompute::MAP_DEVICE_ARRAY) == osgCompute::MAP_DEVICE_ARRAY )
        {
            ////////////////////
            // UNREGISTER TEX //
            ////////////////////
            if( memory._graphicsArray != NULL )
            {
                cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::WARN)
                        << "[osgCuda::TextureBuffer::setup()]: error during cudaGraphicsUnmapResources(). "
                        << cudaGetErrorString( res ) <<"."
                        << std::endl;
                    return false;
                }

                memory._graphicsArray = NULL;
            }


            if( memory._graphicsResource != NULL )
            {
                cudaError res = cudaGraphicsUnregisterResource( memory._graphicsResource );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::setup()]: unable to unregister buffer object. "
                        << std::endl;

                    return false;
                }
                memory._graphicsResource = NULL;
            }

            ////////////////
            // UPDATE TEX //
            ////////////////
            osg::State* state;
            osg::GraphicsContext::GraphicsContexts _ctxs = osg::GraphicsContext::getAllRegisteredGraphicsContexts();
            for( osg::GraphicsContext::GraphicsContexts::iterator itr = _ctxs.begin(); itr != _ctxs.end(); ++itr )
            {
                if( (*itr)->getState() && ((*itr)->getState()->getContextID() == osgCompute::Resource::getCurrentIdx()) )
                {
                    state = (*itr)->getState();
                    break;
                }
            }

            if( NULL == state )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::setup()]: unable to find valid state."
                    << std::endl;

                return false;
            }

            // setup current state
            _texref->apply( *state );

            //////////////////
            // REGISTER TEX //
            //////////////////
            osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
            cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::setup()]: unable to register buffer object again."
                    << std::endl;

                return false;
            }

            memory._syncOp |= osgCompute::SYNC_DEVICE;
            memory._syncOp |= osgCompute::SYNC_HOST;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
			memory._lastModifiedAddress = (unsigned int) _texref->getImage(0);
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            void* data = _texref->getImage(0)->data();
            if( data == NULL )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::setup()]: cannot receive valid data pointer from image."
                    << std::endl;

                return false;
            }

            if( getNumDimensions() == 3 )
            {
                cudaMemcpy3DParms memCpyParams = {0};
                memCpyParams.extent = make_cudaExtent( getDimension(0) * getElementSize(), getDimension(1), getDimension(2) );
                memCpyParams.dstPtr = make_cudaPitchedPtr( memory._devPtr, memory._pitch, getDimension(0), getDimension(1) );
                memCpyParams.kind = cudaMemcpyHostToDevice;
                memCpyParams.srcPtr = make_cudaPitchedPtr( data, memory._pitch, getDimension(0), getDimension(1) );
                cudaError res = cudaMemcpy3D( &memCpyParams );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::setup()]: error during cudaMemcpy()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                cudaError res = cudaMemcpy( memory._devPtr, data, getByteSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::setup()]: error during cudaMemcpy()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }

            // device must be synchronized
            memory._syncOp |= osgCompute::SYNC_HOST;
            memory._syncOp |= osgCompute::SYNC_ARRAY;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
			memory._lastModifiedAddress = (unsigned int) _texref->getImage(0);
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            const void* data = _texref->getImage(0)->data();

            if( data == NULL )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::setup()]: cannot receive valid data pointer from image."
                    << std::endl;

                return false;
            }

            cudaError res = cudaMemcpy( memory._hostPtr, data, getByteSize(), cudaMemcpyHostToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::setup()]: error during cudaMemcpy()."
                    << " " << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            // device must be synchronized
            memory._syncOp |= osgCompute::SYNC_DEVICE;
            memory._syncOp |= osgCompute::SYNC_ARRAY;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
			memory._lastModifiedAddress = (unsigned int) _texref->getImage(0);
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool TextureBuffer::alloc( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object() );
        if( !memoryPtr )
            return NULL;
        TextureObject& memory = *memoryPtr;

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
                    << getName() << " [osgCuda::TextureBuffer::alloc()]: error during mallocHost()."
                    << std::endl;

                return false;
            }

            return true;
        }
        else if( (mapping & osgCompute::MAP_DEVICE_ARRAY) == osgCompute::MAP_DEVICE_ARRAY )
        {
            if( memory._graphicsResource != NULL )
                return true;

            osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
            if( !tex )
            {
                osg::State* state;
                osg::GraphicsContext::GraphicsContexts _ctxs = osg::GraphicsContext::getAllRegisteredGraphicsContexts();
                for( osg::GraphicsContext::GraphicsContexts::iterator itr = _ctxs.begin(); itr != _ctxs.end(); ++itr )
                {
                    if( (*itr)->getState() && ((*itr)->getState()->getContextID() == osgCompute::Resource::getCurrentIdx()) )
                    {
                        state = (*itr)->getState();
                        break;
                    }
                }

                if( NULL == state )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to find valid state."
                        << std::endl;

                    return false;
                }

                _texref->compileGLObjects( *state );
                tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
            }

            // Register vertex buffer object for Cuda
            cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to register image object (cudaGraphicsGLRegisterImage()). Not all GL formats are supported."
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {            
            if( memory._devPtr != NULL )
                return true;

            // Allocate shadow-copy memory
            if( getNumDimensions() == 3 )
            {
                cudaExtent ext = {0};
                ext.width = getDimension(0) * getElementSize();
                ext.height = getDimension(1);
                ext.depth = getDimension(2);

                cudaPitchedPtr pitchPtr;
                cudaError res = cudaMalloc3D( &pitchPtr, ext );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to alloc shadow-copy (cudaMalloc())."
                        << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }

                memory._pitch = pitchPtr.pitch;
                memory._devPtr = pitchPtr.ptr;
            }
            else if( getNumDimensions() == 2 )
            {
                cudaError res = cudaMallocPitch( &memory._devPtr, (size_t*)(&memory._pitch), getDimension(0) * getElementSize(), getDimension(1) );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to alloc shadow-copy (cudaMallocPitch())."
                        << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                cudaError res = cudaMalloc( &memory._devPtr, getByteSize() );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::alloc()]: unable to alloc shadow-copy (cudaMalloc())."
                        << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }

                memory._pitch = getDimension(0) * getElementSize();
            }

            if( memory._pitch != (getDimension(0) * getElementSize()) )
            {
                int device = 0;
                cudaGetDevice( &device );
                int textureAlignment = 0;
                cudaDeviceProp devProp;
                cudaGetDeviceProperties( &devProp, device );

                osg::notify(osg::INFO)
                    << getName() << " [osgCuda::TextureBuffer::alloc()] \""<<getName()
                    << "\": Memory requirement is not a multiple of texture alignment. This "
                    << "leads to a pitch which is not equal to the logical row size in bytes. "
                    << "Texture alignment requirement is \""
                    << devProp.textureAlignment << "\". "
                    << std::endl;
            }

            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool TextureBuffer::sync( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object() );
        if( !memoryPtr )
            return NULL;
        TextureObject& memory = *memoryPtr;

        /////////////////
        // SYNC MEMORY //
        /////////////////
        if( (mapping & osgCompute::MAP_DEVICE_ARRAY) == osgCompute::MAP_DEVICE_ARRAY )
        {
            if( !(memory._syncOp & osgCompute::SYNC_ARRAY) )
                return true;

            if( ((memory._syncOp & osgCompute::SYNC_DEVICE) && memory._hostPtr == NULL) ||
                ((memory._syncOp & osgCompute::SYNC_HOST) && memory._devPtr == NULL) ||
                ((memory._syncOp & osgCompute::SYNC_DEVICE) && (memory._syncOp & osgCompute::SYNC_HOST)) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::sync()]: no current memory found."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_DEVICE) )
            {
                // Copy from host memory
                if( getNumDimensions() < 2 )
                {
                    res = cudaMemcpyToArray( memory._graphicsArray, 0, 0, memory._hostPtr, getByteSize(), cudaMemcpyHostToDevice);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: cudaMemcpyToArray() failed."
                            << " " << cudaGetErrorString( res ) << "."
                            << std::endl;

                        return false;
                    }
                }
                if( getNumDimensions() == 2 ) 
                {
                    res = cudaMemcpy2DToArray( memory._graphicsArray, 0, 0, memory._hostPtr, 
                        getDimension(0)*getElementSize(), getDimension(0)*getElementSize(), getDimension(1), 
                        cudaMemcpyHostToDevice );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: cudaMemcpy2DToArray() failed."
                            << " " << cudaGetErrorString( res ) << "."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    cudaMemcpy3DParms memCpyParams = {0};
                    memCpyParams.dstArray = memory._graphicsArray;
                    memCpyParams.kind = cudaMemcpyHostToDevice;
                    memCpyParams.srcPtr = make_cudaPitchedPtr(memory._hostPtr, getDimension(0)*getElementSize(), getDimension(0), getDimension(1));

                    cudaExtent arrayExtent = {0};
                    arrayExtent.width = getDimension(0);
                    arrayExtent.height = getDimension(1);
                    arrayExtent.depth = getDimension(2);

                    memCpyParams.extent = arrayExtent;

                    res = cudaMemcpy3D( &memCpyParams );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << "[osgCuda::TextureBuffer::sync()]: cudaMemcpy3D() failed."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
            }
            else
            {
                // Copy from device memory
                if( getNumDimensions() < 2 )
                {
                    res = cudaMemcpyToArray( memory._graphicsArray, 0, 0, memory._devPtr, getByteSize(), cudaMemcpyDeviceToDevice);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: cudaMemcpyToArray() failed."
                            << " " << cudaGetErrorString( res ) << "."
                            << std::endl;

                        return false;
                    }
                }
                else if( getNumDimensions() == 2 ) 
                {
                    res = cudaMemcpy2DToArray( memory._graphicsArray, 0, 0, memory._devPtr, 
                                               memory._pitch,  getDimension(0)*getElementSize(), getDimension(1), 
                                               cudaMemcpyDeviceToDevice );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: cudaMemcpy2DToArray() failed."
                            << " " << cudaGetErrorString( res ) << "."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    cudaMemcpy3DParms memCpyParams = {0};
                    memCpyParams.dstArray = memory._graphicsArray;
                    memCpyParams.kind = cudaMemcpyDeviceToDevice;
                    memCpyParams.srcPtr = make_cudaPitchedPtr(memory._devPtr, memory._pitch, getDimension(0), getDimension(1));

                    cudaExtent arrayExtent = {0};
                    arrayExtent.width = getDimension(0);
                    arrayExtent.height = getDimension(1);
                    arrayExtent.depth = getDimension(2);

                    memCpyParams.extent = arrayExtent;

                    res = cudaMemcpy3D( &memCpyParams );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << "[osgCuda::TextureBuffer::sync()]: cudaMemcpy3D() failed."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
            }

            memory._syncOp = memory._syncOp ^ osgCompute::SYNC_ARRAY;
            return true;
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            if( !(memory._syncOp & osgCompute::SYNC_DEVICE) )
                return true;

            if( ((memory._syncOp & osgCompute::SYNC_ARRAY) && memory._hostPtr == NULL) ||
                ((memory._syncOp & osgCompute::SYNC_ARRAY) && (memory._syncOp & osgCompute::SYNC_HOST)) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::sync()]: no current memory found."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_HOST) )
            {
                if( memory._graphicsResource == NULL )
                {
                    osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
                    if( !tex )
                    {
                        osg::notify(osg::WARN)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: no current memory found. "
                            << std::endl;

                        return false;
                    }

                    // Register vertex buffer object for Cuda
                    cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: unable to register image object (cudaGraphicsGLRegisterImage())."
                            << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }

                if( memory._graphicsArray == NULL )
                {
                    // map array first
                    cudaError res = cudaGraphicsMapResources(1, &memory._graphicsResource);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::WARN)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaGraphicsMapResources(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }

                    res = cudaGraphicsSubResourceGetMappedArray( &memory._graphicsArray, memory._graphicsResource, 0, 0);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::WARN)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaGraphicsResourceGetMappedPointer(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }
                }

                // Copy from array
                if( getNumDimensions() == 1 )
                {
                    res = cudaMemcpyFromArray( memory._devPtr, memory._graphicsArray, 0, 0, getByteSize(), cudaMemcpyDeviceToDevice );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]:  error during cudaMemcpyFromArray() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else if( getNumDimensions() == 2 )
                {
                    res = cudaMemcpy2DFromArray(
                        memory._devPtr,
                        memory._pitch,
                        memory._graphicsArray,
                        0, 0,
                        getDimension(0)* getElementSize(),
                        getDimension(1),
                        cudaMemcpyDeviceToDevice );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy2DFromArray() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    cudaPitchedPtr pitchPtr = {0};
                    pitchPtr.pitch = memory._pitch;
                    pitchPtr.ptr = memory._devPtr;
                    pitchPtr.xsize = getDimension(0);
                    pitchPtr.ysize = getDimension(1);

                    cudaExtent extent = {0};
                    extent.width = getDimension(0);
                    extent.height = getDimension(1);
                    extent.depth = getDimension(2);

                    cudaMemcpy3DParms copyParams = {0};
                    copyParams.srcArray = memory._graphicsArray;
                    copyParams.dstPtr = pitchPtr;
                    copyParams.extent = extent;
                    copyParams.kind = cudaMemcpyDeviceToDevice;

                    res = cudaMemcpy3D( &copyParams );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy3D() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;

                    }
                }
            }
            else 
            {
                // Copy from host memory
                res = cudaMemcpy( memory._devPtr, memory._hostPtr, getByteSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy() to device from host. "
                        << cudaGetErrorString( res ) <<"."
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

            if( ((memory._syncOp & osgCompute::SYNC_ARRAY) && memory._devPtr == NULL) ||
                ((memory._syncOp & osgCompute::SYNC_ARRAY) && (memory._syncOp & osgCompute::SYNC_DEVICE)) )
            {
                osg::notify(osg::FATAL)
                    << getName() << " [osgCuda::TextureBuffer::sync()]: no current memory found."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_DEVICE) )
            {
                if( memory._graphicsResource == NULL )
                {
                    osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::Resource::getCurrentIdx() );
                    if( !tex )
                    {
                        osg::notify(osg::WARN)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: no current memory found. "
                            << std::endl;

                        return false;
                    }

                    // Register vertex buffer object for Cuda
                    cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: unable to register image object (cudaGraphicsGLRegisterImage())."
                            << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }

                if( memory._graphicsArray == NULL )
                {
                    // map array first
                    cudaError res = cudaGraphicsMapResources(1, &memory._graphicsResource);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::WARN)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaGraphicsMapResources(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }

                    res = cudaGraphicsSubResourceGetMappedArray( &memory._graphicsArray, memory._graphicsResource, 0, 0);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::WARN)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaGraphicsResourceGetMappedPointer(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }
                }

                // Copy from array
                if( getNumDimensions() == 1 )
                {
                    res = cudaMemcpyFromArray( memory._hostPtr, memory._graphicsArray, 0, 0, getByteSize(), cudaMemcpyDeviceToHost );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]:  error during cudaMemcpyFromArray() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else if( getNumDimensions() == 2 )
                {
                    res = cudaMemcpy2DFromArray(
                        memory._hostPtr,
                        getDimension(0) * getElementSize(),
                        memory._graphicsArray,
                        0, 0,
                        getDimension(0)*getElementSize(),
                        getDimension(1),
                        cudaMemcpyDeviceToHost );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy2DFromArray() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    cudaPitchedPtr pitchPtr = {0};
                    pitchPtr.pitch = getDimension(0)*getElementSize();
                    pitchPtr.ptr = memory._hostPtr;
                    pitchPtr.xsize = getDimension(0);
                    pitchPtr.ysize = getDimension(1);

                    cudaExtent extent = {0};
                    extent.width = getDimension(0);
                    extent.height = getDimension(1);
                    extent.depth = getDimension(2);

                    cudaMemcpy3DParms copyParams = {0};
                    copyParams.srcArray = memory._graphicsArray;
                    copyParams.dstPtr = pitchPtr;
                    copyParams.extent = extent;
                    copyParams.kind = cudaMemcpyDeviceToHost;

                    res = cudaMemcpy3D( &copyParams );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy3D() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;

                    }
                }
            }
            else
            {
                // Copy from device ptr
                if( getNumDimensions() == 3 )
                {
                    cudaPitchedPtr pitchDstPtr = {0};
                    pitchDstPtr.pitch = getDimension(0)*getElementSize();
                    pitchDstPtr.ptr = memory._hostPtr;
                    pitchDstPtr.xsize = getDimension(0);
                    pitchDstPtr.ysize = getDimension(1);

                    cudaPitchedPtr pitchSrcPtr = {0};
                    pitchSrcPtr.pitch = memory._pitch;
                    pitchSrcPtr.ptr = memory._devPtr;
                    pitchSrcPtr.xsize = getDimension(0);
                    pitchSrcPtr.ysize = getDimension(1);

                    cudaExtent extent = {0};
                    extent.width = getDimension(0)*getElementSize();
                    extent.height = getDimension(1);
                    extent.depth = getDimension(2);

                    cudaMemcpy3DParms copyParams = {0};
                    copyParams.srcPtr = pitchSrcPtr;
                    copyParams.dstPtr = pitchDstPtr;
                    copyParams.extent = extent;
                    copyParams.kind = cudaMemcpyDeviceToHost;

                    res = cudaMemcpy3D( &copyParams );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy() to device from host. "
                            << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else if( getNumDimensions() == 2 )
                {
                    res = cudaMemcpy2D( memory._hostPtr, getDimension(0)*getElementSize(), memory._devPtr, memory._pitch, getDimension(0)*getElementSize(), getDimension(1), cudaMemcpyDeviceToHost );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy() to device from host. "
                            << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    res = cudaMemcpy( memory._hostPtr, memory._devPtr, getByteSize(), cudaMemcpyDeviceToHost );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << getName() << " [osgCuda::TextureBuffer::sync()]: error during cudaMemcpy() to device from host. "
                            << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
            }

            memory._syncOp = memory._syncOp ^ osgCompute::SYNC_HOST;
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    osgCompute::MemoryObject* TextureBuffer::createObject() const
    {
        return new TextureObject;
    }




    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ////------------------------------------------------------------------------------
    //Texture1D::Texture1D()
    //    : osg::Texture1D(),
    //    _proxy(NULL)
    //{
    //    clearLocal();

    //    // some flags for textures are not available right now
    //    // like resize to a power of two and mip-maps
    //    asTexture()->setResizeNonPowerOfTwoHint( false );
    //    asTexture()->setUseHardwareMipMapGeneration( false );
    //}

    ////------------------------------------------------------------------------------
    //bool Texture1D::init()
    //{
    //    if( NULL != _proxy )
    //    {
    //        _proxy->_usage = getUsage();
    //        _proxy->init();
    //    }
    //    _clear = false;
    //    return true;
    //}

    ////------------------------------------------------------------------------------
    //osgCompute::InteropMemory* Texture1D::getMemory()
    //{
    //    return _proxy;
    //}

    ////------------------------------------------------------------------------------
    //const osgCompute::InteropMemory* Texture1D::getMemory() const
    //{
    //    return _proxy;
    //}

    ////------------------------------------------------------------------------------
    //osgCompute::InteropMemory* Texture1D::getOrCreateMemory()
    //{
    //    // create proxy buffer on demand
    //    if( _proxy == NULL )
    //    {
    //        _proxy = new TextureBuffer;
    //        _proxy->setName( getName() );
    //        _proxy->_texref = this;
    //        _proxy->_usage = getUsage();
    //        _proxy->setIdentifiers( _identifiers );
    //        _identifiers.clear();
    //        if( !_proxy->init() )
    //        {
    //            _proxy->unref();
    //            _proxy = NULL;
    //        }
    //    }

    //    return _proxy;
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::addIdentifier( const std::string& identifier )
    //{
    //    if( _proxy != NULL )
    //    {
    //        _proxy->addIdentifier( identifier );
    //    }
    //    else
    //    {
    //        if( !isIdentifiedBy(identifier) )
    //            _identifiers.insert( identifier );
    //    }
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::removeIdentifier( const std::string& identifier )
    //{
    //    if( _proxy != NULL )
    //    {
    //        _proxy->removeIdentifier( identifier );
    //    }
    //    else
    //    {
    //        osgCompute::IdentifierSetItr itr = _identifiers.find( identifier );
    //        if( itr != _identifiers.end() )
    //            _identifiers.erase( itr );

    //    }
    //}

    ////------------------------------------------------------------------------------
    //bool Texture1D::isIdentifiedBy( const std::string& identifier ) const
    //{
    //    if( _proxy != NULL )
    //    {
    //        return _proxy->isIdentifiedBy( identifier );
    //    }
    //    else
    //    {
    //        osgCompute::IdentifierSetCnstItr itr = _identifiers.find( identifier );
    //        if( itr == _identifiers.end() )
    //            return false;

    //        return true;
    //    }
    //}

	////------------------------------------------------------------------------------
	//osgCompute::IdentifierSet& Texture1D::getIdentifiers()
	//{
	//	if( _proxy != NULL )
	//	{
	//		return _proxy->getIdentifiers();
	//	}
	//	else
	//	{
	//		return _identifiers;
	//	}
	//}

	////------------------------------------------------------------------------------
	//const osgCompute::IdentifierSet& Texture1D::getIdentifiers() const
	//{
	//	if( _proxy != NULL )
	//	{
	//		return _proxy->getIdentifiers();
	//	}
	//	else
	//	{
	//		return _identifiers;
	//	}
	//}

    ////------------------------------------------------------------------------------
    //void Texture1D::releaseGLObjects( osg::State* state/*=0*/ ) const
    //{
    //    _proxy->clearCurrent();
    //    osg::Texture1D::releaseGLObjects( state );
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::compileGLObjects(osg::State& state) const
    //{
    //    osg::Texture::apply(state);
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::apply(osg::State& state) const
    //{
    //    _proxy->unmap();
    //    osg::Texture1D::apply( state );
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::freeProxy()
    //{
    //    // attach identifiers
    //    _identifiers = _proxy->getIdentifiers();
    //    // proxy is now deleted
    //    _proxy = NULL;
    //}

    ////------------------------------------------------------------------------------
    //bool Texture1D::isClear()
    //{
    //    return _clear;
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::clear()
    //{
    //    clearLocal();
    //}

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ////------------------------------------------------------------------------------
    //Texture1D::~Texture1D()
    //{
    //    if( _proxy != NULL )
    //    {
    //        osg::notify(osg::FATAL)
    //            << getName() << " [osgCuda::Texture1D::destructor()]: proxy is still valid!!!."
    //            << std::endl;
    //    }
    //}

    ////------------------------------------------------------------------------------
    //void Texture1D::clearLocal()
    //{
    //    _clear = true;
    //    if( NULL != _proxy )
    //    {
    //        _proxy->clear();
    //    }
    //    _identifiers.clear();
    //}


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture2D::Texture2D()
        : osg::Texture2D(),
        _proxy(NULL)
    {
        clearLocal();

        // some flags for textures are not available right now
        // like resize to a power of two and mip-maps
        asTexture()->setResizeNonPowerOfTwoHint( false );
        asTexture()->setUseHardwareMipMapGeneration( false );
    }

    //------------------------------------------------------------------------------
    bool Texture2D::init()
    {
        if( NULL != _proxy )
        {
            _proxy->_usage = getUsage();
            _proxy->init();
        }

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* Texture2D::getInteropMemory()
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    const osgCompute::InteropMemory* Texture2D::getInteropMemory() const
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* Texture2D::getOrCreateInteropMemory()
    {
        // create proxy buffer on demand
        if( _proxy == NULL )
        {
            _proxy = new TextureBuffer;
            _proxy->setName( getName() );
            _proxy->_texref = this;
            _proxy->_usage = getUsage();
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
    void Texture2D::addIdentifier( const std::string& identifier )
    {
        if( _proxy != NULL )
        {
            _proxy->addIdentifier( identifier );
        }
        else
        {
            if( !isIdentifiedBy(identifier) )
                _identifiers.insert( identifier );
        }
    }

    //------------------------------------------------------------------------------
    void Texture2D::removeIdentifier( const std::string& identifier )
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
    bool Texture2D::isIdentifiedBy( const std::string& identifier ) const
    {
        if( _proxy != NULL )
        {
            return _proxy->isIdentifiedBy( identifier );
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
	osgCompute::IdentifierSet& Texture2D::getIdentifiers()
	{
		if( _proxy != NULL )
		{
			return _proxy->getIdentifiers();
		}
		else
		{
			return _identifiers;
		}
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& Texture2D::getIdentifiers() const
	{
		if( _proxy != NULL )
		{
			return _proxy->getIdentifiers();
		}
		else
		{
			return _identifiers;
		}
	}

    //------------------------------------------------------------------------------
    void Texture2D::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        if( _proxy != NULL )
            _proxy->clearCurrent();
        osg::Texture2D::releaseGLObjects( state );
    }

    //------------------------------------------------------------------------------
    void Texture2D::compileGLObjects(osg::State& state) const
    {
        osg::Texture2D::apply(state);
    }

    //------------------------------------------------------------------------------
    void Texture2D::apply(osg::State& state) const
    {
        if( _proxy != NULL )
            _proxy->unmap();
        osg::Texture2D::apply( state );
    }

    //------------------------------------------------------------------------------
    void Texture2D::freeProxy()
    {
        if( _proxy != NULL )
        {
            // attach identifiers
            _identifiers = _proxy->getIdentifiers();
            // proxy is now deleted
            _proxy = NULL;
        }

    }

    //------------------------------------------------------------------------------
    bool Texture2D::isClear()
    {
        return _clear;
    }

    //------------------------------------------------------------------------------
    void Texture2D::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture2D::~Texture2D()
    {
        if( _proxy != NULL )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::Texture2D::destructor()]: proxy is still valid!!!."
                << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void Texture2D::clearLocal()
    {
        _clear = true;
        if( NULL != _proxy )
        {
            _proxy->clear();
        }
        _identifiers.clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture3D::Texture3D()
        : osg::Texture3D(),
        _proxy(NULL)
    {
        clearLocal();

        // some flags for textures are not available right now
        // like resize to a power of two and mip-maps
        asTexture()->setResizeNonPowerOfTwoHint( false );
        asTexture()->setUseHardwareMipMapGeneration( false );
    }

    //------------------------------------------------------------------------------
    bool Texture3D::init()
    {
        if( NULL != _proxy )
        {
            _proxy->_usage = getUsage();
            _proxy->init();
        }

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* Texture3D::getInteropMemory()
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    const osgCompute::InteropMemory* Texture3D::getInteropMemory() const
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* Texture3D::getOrCreateInteropMemory()
    {
        // create proxy buffer on demand
        if( _proxy == NULL )
        {
            _proxy = new TextureBuffer;
            _proxy->setName( getName() );
            _proxy->_texref = this;
            _proxy->_usage = getUsage();
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
    void Texture3D::addIdentifier( const std::string& identifier )
    {
        if( _proxy != NULL )
        {
            _proxy->addIdentifier( identifier );
        }
        else
        {
            if( !isIdentifiedBy(identifier) )
                _identifiers.insert( identifier );
        }
    }

    //------------------------------------------------------------------------------
    void Texture3D::removeIdentifier( const std::string& identifier )
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
    bool Texture3D::isIdentifiedBy( const std::string& identifier ) const
    {
        if( _proxy != NULL )
        {
            return _proxy->isIdentifiedBy( identifier );
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
	osgCompute::IdentifierSet& Texture3D::getIdentifiers()
	{
		if( _proxy != NULL )
		{
			return _proxy->getIdentifiers();
		}
		else
		{
			return _identifiers;
		}
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& Texture3D::getIdentifiers() const
	{
		if( _proxy != NULL )
		{
			return _proxy->getIdentifiers();
		}
		else
		{
			return _identifiers;
		}
	}

    //------------------------------------------------------------------------------
    void Texture3D::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        if( _proxy != NULL )
            _proxy->clearCurrent();
        osg::Texture3D::releaseGLObjects( state );
    }

    //------------------------------------------------------------------------------
    void Texture3D::compileGLObjects(osg::State& state) const
    {
        osg::Texture3D::apply(state);
    }

    //------------------------------------------------------------------------------
    void Texture3D::apply(osg::State& state) const
    {
        if( _proxy != NULL )
            _proxy->unmap();
        osg::Texture3D::apply( state );
    }

    //------------------------------------------------------------------------------
    void Texture3D::freeProxy()
    {
        if( _proxy != NULL )
        {
            // attach identifiers
            _identifiers = _proxy->getIdentifiers();
            // proxy is now deleted
            _proxy = NULL;
        }
    }

    //------------------------------------------------------------------------------
    bool Texture3D::isClear()
    {
        return _clear;
    }

    //------------------------------------------------------------------------------
    void Texture3D::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture3D::~Texture3D()
    {
        if( _proxy != NULL )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::Texture3D::destructor()]: proxy is still valid!!!."
                << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void Texture3D::clearLocal()
    {
        _clear = true;
        if( NULL != _proxy )
        {
            _proxy->clear();
        }
        _identifiers.clear();
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    TextureRectangle::TextureRectangle()
        : osg::TextureRectangle(),
        _proxy(NULL)
    {
        clearLocal();

        // some flags for textures are not available right now
        // like resize to a power of two and mip-maps
        asTexture()->setResizeNonPowerOfTwoHint( false );
        asTexture()->setUseHardwareMipMapGeneration( false );
    }

    //------------------------------------------------------------------------------
    bool TextureRectangle::init()
    {
        if( NULL != _proxy )
        {
            _proxy->_usage = getUsage();
            _proxy->init();
        }

        _clear = false;
        return true;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* TextureRectangle::getInteropMemory()
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    const osgCompute::InteropMemory* TextureRectangle::getInteropMemory() const
    {
        return _proxy;
    }

    //------------------------------------------------------------------------------
    osgCompute::InteropMemory* TextureRectangle::getOrCreateInteropMemory()
    {
        // create proxy buffer on demand
        if( _proxy == NULL )
        {
            _proxy = new TextureBuffer;
            _proxy->setName( getName() );
            _proxy->_texref = this;
            _proxy->_usage = getUsage();
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
    void TextureRectangle::addIdentifier( const std::string& identifier )
    {
        if( _proxy != NULL )
        {
            _proxy->addIdentifier( identifier );
        }
        else
        {
            if( !isIdentifiedBy(identifier) )
                _identifiers.insert( identifier );
        }
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::removeIdentifier( const std::string& identifier )
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
    bool TextureRectangle::isIdentifiedBy( const std::string& identifier ) const
    {
        if( _proxy != NULL )
        {
            return _proxy->isIdentifiedBy( identifier );
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
	osgCompute::IdentifierSet& TextureRectangle::getIdentifiers()
	{
		if( _proxy != NULL )
		{
			return _proxy->getIdentifiers();
		}
		else
		{
			return _identifiers;
		}
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& TextureRectangle::getIdentifiers() const
	{
		if( _proxy != NULL )
		{
			return _proxy->getIdentifiers();
		}
		else
		{
			return _identifiers;
		}
	}

    //------------------------------------------------------------------------------
    void TextureRectangle::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        if( _proxy != NULL )
            _proxy->clearCurrent();
        osg::TextureRectangle::releaseGLObjects( state );
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::compileGLObjects(osg::State& state) const
    {
        osg::TextureRectangle::apply(state);
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::apply(osg::State& state) const
    {
        if( _proxy != NULL )
            _proxy->unmap();
        osg::TextureRectangle::apply( state );
    }

    //------------------------------------------------------------------------------
    bool TextureRectangle::isClear()
    {
        return _clear;
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::freeProxy()
    {
        if( _proxy != NULL )
        {
            // attach identifiers
            _identifiers = _proxy->getIdentifiers();
            // proxy is now deleted
            _proxy = NULL;
        }
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    TextureRectangle::~TextureRectangle()
    {
        if( _proxy != NULL )
        {
            osg::notify(osg::FATAL)
                << getName() << " [osgCuda::TextureRectangle::destructor()]: proxy is still valid!!!."
                << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::clearLocal()
    {
        _clear = true;
        if( NULL != _proxy )
        {
            _proxy->clear();
        }
        _identifiers.clear();
    }
}
