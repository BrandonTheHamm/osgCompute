#include <memory.h>
#if defined(__linux)
#include <malloc.h>
#endif
#include <osg/GL>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <osg/observer_ptr>
#include <osgCompute/Memory>
#include <osgCuda/Texture>

namespace osgCuda
{
    /**
    */
    class LIBRARY_EXPORT TextureObject : public osgCompute::MemoryObject
    {
    public:
        void*						_hostPtr;
        void*						_devPtr;
        cudaArray*                  _graphicsArray;
        cudaGraphicsResource*       _graphicsResource;
        unsigned int	            _lastModifiedCount;
		void*						_lastModifiedAddress;

        TextureObject();
        virtual ~TextureObject();


    private:
        // not allowed to call copy-constructor or copy-operator
        TextureObject( const TextureObject& ) {}
        TextureObject& operator=( const TextureObject& ) { return *this; }
    };

    /**
    */
    class LIBRARY_EXPORT TextureMemory : public osgCompute::GLMemory
    {
    public:
        TextureMemory();

        META_Object(osgCuda,TextureMemory)

		virtual osgCompute::GLMemoryAdapter* getAdapter(); 
		virtual const osgCompute::GLMemoryAdapter* getAdapter() const; 

        virtual unsigned int getElementSize() const;
        virtual unsigned int getDimension( unsigned int dimIdx ) const;
        virtual unsigned int getNumDimensions() const;
        virtual unsigned int getNumElements() const;

        virtual void* map( unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int offset = 0, unsigned int hint = 0 );
        virtual void unmap( unsigned int hint = 0 );
        virtual bool reset( unsigned int hint = 0 );
        virtual bool supportsMapping( unsigned int mapping, unsigned int hint = 0 ) const;
        virtual void mapAsRenderTarget();
        virtual unsigned int getAllocatedByteSize( unsigned int mapping, unsigned int hint = 0 ) const;
        virtual unsigned int getByteSize( unsigned int mapping = osgCompute::MAP_DEVICE, unsigned int hint = 0 ) const;

    protected:
        friend class Texture1D;
        friend class Texture2D;
        friend class Texture3D;
        friend class TextureRectangle;
        virtual ~TextureMemory();

        bool setup( unsigned int mapping );
        bool alloc( unsigned int mapping );
        bool sync( unsigned int mapping );

        virtual osgCompute::MemoryObject* createObject() const;
        virtual unsigned int computePitch() const;

        osg::observer_ptr<osg::Texture>	_texref; 
    private:
        // copy constructor and operator should not be called
        TextureMemory( const TextureMemory& , const osg::CopyOp& ) {}
        TextureMemory& operator=(const TextureMemory&) { return (*this); }
    };

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
		  _lastModifiedAddress(NULL)
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
                <<__FUNCTION__ <<": error during cudaFree()."
                << cudaGetErrorString(res) << std::endl;
            }
        }

        if( _graphicsArray != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &_graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<": error during cudaGLUnmapBufferObject(). "
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
                << __FUNCTION__ <<": error during cudaGraphicsUnregisterResource()."
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
    TextureMemory::TextureMemory()
		: osgCompute::GLMemory()
    {
        // Please note that virtual functions className() and libraryName() are called
        // during observeResource() which will only develop until this class.
        // However if contructor of a subclass calls this function again observeResource
        // will change the className and libraryName of the observed pointer.
        osgCompute::ResourceObserver::instance()->observeResource( *this );
    }

    //------------------------------------------------------------------------------
    TextureMemory::~TextureMemory()
    {
    }

	//------------------------------------------------------------------------------
	osgCompute::GLMemoryAdapter* TextureMemory::getAdapter()
	{ 
		return dynamic_cast<osgCompute::GLMemoryAdapter*>( _texref.get() );
	}

	//------------------------------------------------------------------------------
	const osgCompute::GLMemoryAdapter* TextureMemory::getAdapter() const
	{ 
		return dynamic_cast<const osgCompute::GLMemoryAdapter*>( _texref.get() );
	}

    //------------------------------------------------------------------------------
    unsigned int TextureMemory::getElementSize() const 
    { 
        unsigned int elementSize = osgCompute::Memory::getElementSize();
        if( osgCompute::Memory::getElementSize() == 0 )
        {
            if( !_texref.valid() )
                return 0;

            elementSize = 0;
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
            const_cast<osgCuda::TextureMemory*>(this)->setElementSize( elementSize );
        }

        return elementSize; 
    }

    //------------------------------------------------------------------------------
    unsigned int TextureMemory::getDimension( unsigned int dimIdx ) const
    { 
        if( osgCompute::Memory::getNumDimensions() == 0 )
        {
            if( !_texref.valid() )
                return 0;

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
                return 0;

            unsigned int d = 0;
            while( dim[d] > 1 && d < 3 )
            {
                const_cast<osgCuda::TextureMemory*>(this)->setDimension( d, dim[d] );
                ++d;
            }
        }

        return osgCompute::Memory::getDimension(dimIdx);
    }

    //------------------------------------------------------------------------------
    unsigned int TextureMemory::getNumDimensions() const
    {
        unsigned int numDims = osgCompute::Memory::getNumDimensions();
        if( numDims == 0 )
        {
            if( !_texref.valid() )
                return 0;

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
                return 0;

            unsigned int d = 0;
            while( dim[d] > 1 && d < 3 )
            {
                const_cast<osgCuda::TextureMemory*>(this)->setDimension( d, dim[d] );
                ++d;
            }

            numDims = osgCompute::Memory::getNumDimensions();
        }

        return numDims;
    }

    //------------------------------------------------------------------------------
    unsigned int TextureMemory::getNumElements() const
    {
        unsigned int numElements = osgCompute::Memory::getNumElements();
        if( numElements == 0 )
        {
            if( !_texref.valid() )
                return 0;

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
                return 0;

            unsigned int d = 0;
            while( dim[d] > 1 && d < 3 )
            {
                const_cast<osgCuda::TextureMemory*>(this)->setDimension( d, dim[d] );
                ++d;
            }

            numElements = osgCompute::Memory::getNumElements();
        }

        return numElements;
    }

    //------------------------------------------------------------------------------
     unsigned int TextureMemory::getAllocatedByteSize( unsigned int mapping, unsigned int hint /*= 0 */ ) const
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        const TextureObject* memoryPtr = dynamic_cast<const TextureObject*>( object(false) );
        if( !memoryPtr )
            return NULL;
        const TextureObject& memory = *memoryPtr;

        unsigned int allocSize = 0;
        switch( mapping )
        {
        case osgCompute::MAP_DEVICE: case osgCompute::MAP_DEVICE_TARGET: case osgCompute::MAP_DEVICE_SOURCE:
            {
                allocSize = (memory._devPtr != NULL)? getByteSize( mapping, hint ) : 0;

            }break;
        case osgCompute::MAP_HOST: case osgCompute::MAP_HOST_TARGET: case osgCompute::MAP_HOST_SOURCE:
            {
                allocSize = (memory._hostPtr != NULL)? getByteSize( mapping, hint ) : 0;

            }break;
        case osgCompute::MAP_DEVICE_ARRAY: case osgCompute::MAP_DEVICE_ARRAY_TARGET:
            {
                allocSize = (memory._graphicsResource != NULL)? getByteSize( mapping, hint ) : 0;

            }break;
        }

        return allocSize;
    }

    //------------------------------------------------------------------------------
    unsigned int TextureMemory::getByteSize( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int hint /*= 0 */  ) const
    {
        unsigned int allocSize = 0;
        switch( mapping )
        {
        case osgCompute::MAP_DEVICE: case osgCompute::MAP_DEVICE_TARGET: case osgCompute::MAP_DEVICE_SOURCE:
            {
                unsigned int dimensions = getNumDimensions();
                allocSize = getPitch();

                for( unsigned int d=1; d<dimensions; ++d )
                    allocSize *= getDimension(d);

            }break;
        case osgCompute::MAP_HOST: case osgCompute::MAP_HOST_TARGET: case osgCompute::MAP_HOST_SOURCE:
            {
                allocSize = getElementSize() * getNumElements();

            }break;
        case osgCompute::MAP_DEVICE_ARRAY: case osgCompute::MAP_DEVICE_ARRAY_TARGET:
            {
                unsigned int dimensions = getNumDimensions();
                allocSize = getPitch();

                for( unsigned int d=1; d<dimensions; ++d )
                    allocSize *= getDimension(d);

            }break;
        }

        return allocSize;
    }

    //------------------------------------------------------------------------------
    void* TextureMemory::map( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int hint/* = 0*/ )
    {
		if( !_texref.valid() )
			return NULL;

        if( mapping == osgCompute::UNMAP )
        {
            unmap( hint );
            return NULL;
        }

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(true) );
        if( !memoryPtr )
            return NULL;
        TextureObject& memory = *memoryPtr;

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
			  memory._lastModifiedAddress != _texref->getImage(0) )
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
            if( osgCompute::GLMemory::getContext() == NULL || osgCompute::GLMemory::getContext()->getState() == NULL )
                return NULL;

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
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsMapResources(). "
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    return NULL;
                }

                res = cudaGraphicsSubResourceGetMappedArray( &memory._graphicsArray, memory._graphicsResource, 0, 0);
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
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
                << __FUNCTION__ <<" " << _texref->getName() << ": Wrong mapping type specified. Use one of the following types: "
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

        if( (mapping & osgCompute::MAP_DEVICE_ARRAY_TARGET) == osgCompute::MAP_DEVICE_ARRAY_TARGET )
        {
            memory._syncOp |= osgCompute::SYNC_DEVICE;
            memory._syncOp |= osgCompute::SYNC_HOST;
        }
        else if( (mapping & osgCompute::MAP_DEVICE_TARGET) == osgCompute::MAP_DEVICE_TARGET )
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
    void TextureMemory::unmap( unsigned int )
    {
		if( !_texref.valid() )
			return;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(false) );
        if( !memoryPtr )
            return;
        TextureObject& memory = *memoryPtr;

        //////////////////
        // UNMAP MEMORY //
        //////////////////
        // Copy current memory to texture memory
        if( memory._syncOp & osgCompute::SYNC_ARRAY && osgCompute::GLMemory::getContext() != NULL )
        {
            if( NULL == map( osgCompute::MAP_DEVICE_ARRAY, 0 ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during device memory synchronization (map())."
                    << std::endl;

                return;
            }
        }

        // Change current context to render context
        if( memory._graphicsArray != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            memory._graphicsArray = NULL;
        }

        memory._mapping = osgCompute::UNMAP;
    }

    //------------------------------------------------------------------------------
    bool TextureMemory::reset( unsigned int  )
    {
        if( !_texref.valid()  )
			return false;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(false) );
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
            if( !memset( memory._hostPtr, 0x0, getAllElementsSize() ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during memset() for host memory."
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
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemset3D() for device memory."
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
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemset2D() for device memory."
                        << cudaGetErrorString( res )  <<"."
                        << std::endl;

                    unmap();
                    return false;
                }
            }
            else
            {
                res = cudaMemset( memory._devPtr, 0x0, getAllElementsSize() );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemset() for device memory."
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
    bool TextureMemory::supportsMapping( unsigned int mapping, unsigned int ) const
    {
		if( !_texref.valid() )
			return false;

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

    //------------------------------------------------------------------------------
    void TextureMemory::mapAsRenderTarget()
    {
        if( !_texref.valid() )
            return;

        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(true) );
        if( !memoryPtr )
            return;
        TextureObject& memory = *memoryPtr;

        if( memory._syncOp & osgCompute::SYNC_ARRAY )
        {
            if( NULL == map( osgCompute::MAP_DEVICE_ARRAY, 0 ) )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during device memory synchronization (map())."
                    << std::endl;

                return;
            }
        }

        // Host memory and device memory should be synchronized in next call to map
        memory._syncOp |= osgCompute::SYNC_DEVICE;
        memory._syncOp |= osgCompute::SYNC_HOST;

        if( memory._graphicsArray != NULL )
        {
            cudaError res = cudaGraphicsUnmapResources( 1, &memory._graphicsResource );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGLUnmapBufferObject(). "
                    << cudaGetErrorString( res ) <<"."
                    << std::endl;
                return;
            }
            memory._graphicsArray = NULL;
        }
    }


    
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    unsigned int TextureMemory::computePitch() const
    {
        // Proof paramters
        if( getNumDimensions() == 0 || getElementSize() == 0 ) 
            return 0;

        // 1-dimensional layout
        if ( getNumDimensions() < 2 )
            return getElementSize() * getNumElements();

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
    bool TextureMemory::setup( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(false) );
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
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsUnmapResources(). "
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
                        << __FUNCTION__ <<" " << _texref->getName() << ": unable to unregister buffer object. "
                        << std::endl;

                    return false;
                }
                memory._graphicsResource = NULL;
            }

            ////////////////
            // UPDATE TEX //
            ////////////////
            osg::State* state = osgCompute::GLMemory::getContext()->getState();
            if( NULL == state )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": unable to find valid state."
                    << std::endl;

                return false;
            }

            // setup current state
            _texref->apply( *state );

            //////////////////
            // REGISTER TEX //
            //////////////////
            osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": unable to register buffer object again."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_ARRAY) == osgCompute::SYNC_ARRAY )
                memory._syncOp ^= osgCompute::SYNC_ARRAY;

            memory._syncOp |= osgCompute::SYNC_DEVICE;
            memory._syncOp |= osgCompute::SYNC_HOST;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
			memory._lastModifiedAddress = _texref->getImage(0);
        }
        else if( mapping & osgCompute::MAP_DEVICE )
        {
            void* data = _texref->getImage(0)->data();
            if( data == NULL )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": cannot receive valid data pointer from image."
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
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy3D()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else if( getNumDimensions() == 2) 
            {
                cudaError res = cudaMemcpy2D( memory._devPtr, memory._pitch, data, getDimension(0)*getElementSize(), getDimension(0)*getElementSize(), getDimension(1),  cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy2D()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                cudaError res = cudaMemcpy( memory._devPtr, data, getDimension(0)*getElementSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy()."
                        << " " << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }

            if( (memory._syncOp & osgCompute::SYNC_DEVICE) == osgCompute::SYNC_DEVICE )
                memory._syncOp ^= osgCompute::SYNC_DEVICE;

            // device must be synchronized
            memory._syncOp |= osgCompute::SYNC_HOST;
            memory._syncOp |= osgCompute::SYNC_ARRAY;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
			memory._lastModifiedAddress = _texref->getImage(0);
        }
        else if( mapping & osgCompute::MAP_HOST )
        {
            const void* data = _texref->getImage(0)->data();

            if( data == NULL )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": cannot receive valid data pointer from image."
                    << std::endl;

                return false;
            }

            cudaError res = cudaMemcpy( memory._hostPtr, data, getAllElementsSize(), cudaMemcpyHostToHost );
            if( cudaSuccess != res )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy()."
                    << " " << cudaGetErrorString( res ) <<"."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_HOST) == osgCompute::SYNC_HOST )
                memory._syncOp ^= osgCompute::SYNC_HOST;

            // device must be synchronized
            memory._syncOp |= osgCompute::SYNC_DEVICE;
            memory._syncOp |= osgCompute::SYNC_ARRAY;
            memory._lastModifiedCount = _texref->getImage(0)->getModifiedCount();
			memory._lastModifiedAddress = _texref->getImage(0);
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool TextureMemory::alloc( unsigned int mapping )
    {
        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(true) );
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

            memory._hostPtr = malloc( getAllElementsSize() );
            if( NULL == memory._hostPtr )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ": error during mallocHost()."
                    << std::endl;

                return false;
            }

            return true;
        }
        else if( (mapping & osgCompute::MAP_DEVICE_ARRAY) == osgCompute::MAP_DEVICE_ARRAY )
        {
            if( memory._graphicsResource != NULL )
                return true;

            osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
            if( !tex )
            {
                osg::State* state = osgCompute::GLMemory::getContext()->getState();
                if( NULL == state )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": unable to find valid state."
                        << std::endl;

                    return false;
                }

                _texref->compileGLObjects( *state );
                // unbind texture after compilation
                glBindTexture( _texref->getTextureTarget(), 0 );

                tex = _texref->getTextureObject(osgCompute::GLMemory::getContext()->getState()->getContextID() );
            }

            // Register vertex buffer object for Cuda
            cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
            if( res != cudaSuccess )
            {
                osg::notify(osg::FATAL)
                    << __FUNCTION__ <<" " << _texref->getName() << ":unable to register image object (cudaGraphicsGLRegisterImage()). Not all GL formats are supported."
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
                        << __FUNCTION__ <<" " << _texref->getName() << ":unable to alloc shadow-copy (cudaMalloc())."
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
                        << __FUNCTION__ <<" " << _texref->getName() << ":unable to alloc shadow-copy (cudaMallocPitch())."
                        << cudaGetErrorString( res ) <<"."
                        << std::endl;

                    return false;
                }
            }
            else
            {
                cudaError res = cudaMalloc( &memory._devPtr, getAllElementsSize() );
                if( res != cudaSuccess )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ":unable to alloc shadow-copy (cudaMalloc())."
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
                cudaDeviceProp devProp;
                cudaGetDeviceProperties( &devProp, device );

                osg::notify(osg::INFO)
					<< __FUNCTION__ <<" " << _texref->getName() << ": "
                    << "memory requirement is not a multiple of texture alignment. This "
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
    bool TextureMemory::sync( unsigned int mapping )
    {
        cudaError res;

        ////////////////////
        // RECEIVE HANDLE //
        ////////////////////
        TextureObject* memoryPtr = dynamic_cast<TextureObject*>( object(false) );
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
                    << __FUNCTION__ <<" " << _texref->getName() << ": no current memory found."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_DEVICE) )
            {
                // Copy from host memory
                if( getNumDimensions() < 2 )
                {
                    res = cudaMemcpyToArray( memory._graphicsArray, 0, 0, memory._hostPtr, getAllElementsSize(), cudaMemcpyHostToDevice);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": cudaMemcpyToArray() failed."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": cudaMemcpy2DToArray() failed."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": cudaMemcpy3D() failed."
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
                    res = cudaMemcpyToArray( memory._graphicsArray, 0, 0, memory._devPtr, getAllElementsSize(), cudaMemcpyDeviceToDevice);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": cudaMemcpyToArray() failed."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": cudaMemcpy2DToArray() failed."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": cudaMemcpy3D() failed."
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
                    << __FUNCTION__ <<" " << _texref->getName() << ": no current memory found."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_HOST) )
            {
                if( memory._graphicsResource == NULL )
                {
                    osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
                    if( !tex )
                    {
                        osg::notify(osg::NOTICE)
                            << __FUNCTION__ <<" " << _texref->getName() << ": no current memory found. "
                            << std::endl;

                        return false;
                    }

                    // Register vertex buffer object for Cuda
                    cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": unable to register image object (cudaGraphicsGLRegisterImage())."
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
                        osg::notify(osg::FATAL)
                            <<__FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsMapResources(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }

                    res = cudaGraphicsSubResourceGetMappedArray( &memory._graphicsArray, memory._graphicsResource, 0, 0);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }
                }

                // Copy from array
                if( getNumDimensions() == 3 )
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy3D() to host memory."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy2DFromArray() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    res = cudaMemcpyFromArray( memory._devPtr, memory._graphicsArray, 0, 0, getAllElementsSize(), cudaMemcpyDeviceToDevice );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpyFromArray() to host memory."
                            << " " << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
            }
            else 
            {
                // Copy from host memory
                res = cudaMemcpy( memory._devPtr, memory._hostPtr, getAllElementsSize(), cudaMemcpyHostToDevice );
                if( cudaSuccess != res )
                {
                    osg::notify(osg::FATAL)
                        << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy() to device from host. "
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
                    << __FUNCTION__ <<" " << _texref->getName() << ": no current memory found."
                    << std::endl;

                return false;
            }

            if( (memory._syncOp & osgCompute::SYNC_DEVICE) )
            {
                if( memory._graphicsResource == NULL )
                {
                    osg::Texture::TextureObject* tex = _texref->getTextureObject( osgCompute::GLMemory::getContext()->getState()->getContextID() );
                    if( !tex )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": no current memory found. "
                            << std::endl;

                        return false;
                    }

                    // Register vertex buffer object for Cuda
                    cudaError res = cudaGraphicsGLRegisterImage( &memory._graphicsResource, tex->id(), tex->_profile._target, cudaGraphicsMapFlagsNone );
                    if( res != cudaSuccess )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": unable to register image object (cudaGraphicsGLRegisterImage())."
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
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsMapResources(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }

                    res = cudaGraphicsSubResourceGetMappedArray( &memory._graphicsArray, memory._graphicsResource, 0, 0);
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaGraphicsResourceGetMappedPointer(). "
                            << cudaGetErrorString( res )  <<"."
                            << std::endl;

                        return NULL;
                    }
                }

                // Copy from array
                if( getNumDimensions() == 1 )
                {
                    res = cudaMemcpyFromArray( memory._hostPtr, memory._graphicsArray, 0, 0, getDimension(0)*getElementSize(), cudaMemcpyDeviceToHost );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ":  error during cudaMemcpyFromArray() to host memory."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy2DFromArray() to host memory."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy3D() to host memory."
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy() to device from host. "
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
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy() to device from host. "
                            << cudaGetErrorString( res ) <<"."
                            << std::endl;

                        return false;
                    }
                }
                else
                {
                    res = cudaMemcpy( memory._hostPtr, memory._devPtr, getDimension(0)*getElementSize(), cudaMemcpyDeviceToHost );
                    if( cudaSuccess != res )
                    {
                        osg::notify(osg::FATAL)
                            << __FUNCTION__ <<" " << _texref->getName() << ": error during cudaMemcpy() to device from host. "
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
    osgCompute::MemoryObject* TextureMemory::createObject() const
    {
        return new TextureObject;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture2D::Texture2D()
        : osg::Texture2D(),
		  osgCompute::GLMemoryAdapter()
    {
		TextureMemory* memory = new TextureMemory;
		memory->_texref = this;
		_memory = memory;

        // some flags for textures are not available right now
        // like resize to a power of two and mip-maps
        asTexture()->setResizeNonPowerOfTwoHint( false );
        asTexture()->setUseHardwareMipMapGeneration( false );
        asTexture()->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        asTexture()->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
    }

    //------------------------------------------------------------------------------
    osgCompute::GLMemory* Texture2D::getMemory()
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    const osgCompute::GLMemory* Texture2D::getMemory() const
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    void Texture2D::addIdentifier( const std::string& identifier )
    {
        _memory->addIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    void Texture2D::removeIdentifier( const std::string& identifier )
    {
        _memory->removeIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    bool Texture2D::isIdentifiedBy( const std::string& identifier ) const
    {
        return _memory->isIdentifiedBy( identifier );
    }

	//------------------------------------------------------------------------------
	osgCompute::IdentifierSet& Texture2D::getIdentifiers()
	{
		return _memory->getIdentifiers();
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& Texture2D::getIdentifiers() const
	{
		return _memory->getIdentifiers();
	}

    //------------------------------------------------------------------------------
    void Texture2D::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        // Currently we support  a single OpenGL context only. So clear memory every
        // time releaseGLObjects() is called.
        //if( state != NULL && 
        //    osgCompute::GLMemory::getContext() != NULL && 
        //    state->getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->clear();

        osg::Texture2D::releaseGLObjects( state );
    }

    ////------------------------------------------------------------------------------
    //void Texture2D::resizeGLObjectBuffers( unsigned int maxSize )
    //{
    //    // Currently we support  a single OpenGL context only. So clear memory every
    //    // time releaseGLObjects() is called.
    //    //if( osgCompute::GLMemory::getContext() != NULL )
    //    _memory->clear();

    //    osg::Texture::resizeGLObjectBuffers( maxSize );
    //}

    //------------------------------------------------------------------------------
    void Texture2D::compileGLObjects(osg::State& state) const
    {
        osg::Texture2D::apply(state);
    }

    //------------------------------------------------------------------------------
    void Texture2D::apply(osg::State& state) const
    {
        // Currently we support  a single OpenGL context only. So unmap memory every
        // time releaseGLObjects() is called.
        //if( osgCompute::GLMemory::getContext() != NULL && 
        //    state.getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->unmap();

        osg::Texture2D::apply( state );
    }

    //------------------------------------------------------------------------------
    void Texture2D::applyAsRenderTarget() const
    {
        static_cast<osgCompute::GLMemory*>( _memory )->mapAsRenderTarget();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture2D::~Texture2D()
    {
		// _memory object is not deleted until this point
		// as reference count is increased in constructor
        _memory->releaseObjects();
		_memory = NULL;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture3D::Texture3D()
        : osg::Texture3D(),
		  osgCompute::GLMemoryAdapter()
    {
		TextureMemory* memory = new TextureMemory;
		memory->_texref = this;
		_memory = memory;

        // some flags for textures are not available right now
        // like resize to a power of two and mip-maps
        asTexture()->setResizeNonPowerOfTwoHint( false );
        asTexture()->setUseHardwareMipMapGeneration( false );
        asTexture()->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        asTexture()->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
    }

    //------------------------------------------------------------------------------
    osgCompute::GLMemory* Texture3D::getMemory()
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    const osgCompute::GLMemory* Texture3D::getMemory() const
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    void Texture3D::addIdentifier( const std::string& identifier )
    {
        _memory->addIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    void Texture3D::removeIdentifier( const std::string& identifier )
    {
        _memory->removeIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    bool Texture3D::isIdentifiedBy( const std::string& identifier ) const
    {
        return _memory->isIdentifiedBy( identifier );
    }

	//------------------------------------------------------------------------------
	osgCompute::IdentifierSet& Texture3D::getIdentifiers()
	{
		return _memory->getIdentifiers();
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& Texture3D::getIdentifiers() const
	{
		return _memory->getIdentifiers();
	}

    //------------------------------------------------------------------------------
    void Texture3D::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        // Currently we support  a single OpenGL context only. So clear memory every
        // time releaseGLObjects() is called.
        //if( state != NULL && 
        //    osgCompute::GLMemory::getContext() != NULL && 
        //    state->getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->clear();

        osg::Texture3D::releaseGLObjects( state );
    }

    ////------------------------------------------------------------------------------
    //void Texture3D::resizeGLObjectBuffers( unsigned int maxSize )
    //{
    //    // Currently we support  a single OpenGL context only. So clear memory every
    //    // time releaseGLObjects() is called.
    //    //if( osgCompute::GLMemory::getContext() != NULL )
    //    _memory->clear();

    //    osg::Texture::resizeGLObjectBuffers( maxSize );
    //}

    //------------------------------------------------------------------------------
    void Texture3D::compileGLObjects(osg::State& state) const
    {
        osg::Texture3D::apply(state);
    }

    //------------------------------------------------------------------------------
    void Texture3D::apply(osg::State& state) const
    {
        // Currently we support  a single OpenGL context only. So unmap memory every
        // time releaseGLObjects() is called.
        //if( osgCompute::GLMemory::getContext() != NULL && 
        //    state.getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->unmap();

        osg::Texture3D::apply( state );
    }

    //------------------------------------------------------------------------------
    void Texture3D::applyAsRenderTarget() const
    {
        static_cast<osgCompute::GLMemory*>( _memory )->mapAsRenderTarget();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Texture3D::~Texture3D()
    {
		// _memory object is not deleted until this point
		// as reference count is increased in constructor
        _memory->releaseObjects();
		_memory = NULL;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    TextureRectangle::TextureRectangle()
        : osg::TextureRectangle(),
		  osgCompute::GLMemoryAdapter()
    {
		TextureMemory* memory = new TextureMemory;
		memory->_texref = this;
		_memory = memory;

        // some flags for textures are not available right now
        // like resize to a power of two and mip-maps
        asTexture()->setResizeNonPowerOfTwoHint( false );
        asTexture()->setUseHardwareMipMapGeneration( false );
        asTexture()->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        asTexture()->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
    }

    //------------------------------------------------------------------------------
    osgCompute::GLMemory* TextureRectangle::getMemory()
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    const osgCompute::GLMemory* TextureRectangle::getMemory() const
    {
        return _memory;
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::addIdentifier( const std::string& identifier )
    {
        _memory->addIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::removeIdentifier( const std::string& identifier )
    {
        _memory->removeIdentifier( identifier );
    }

    //------------------------------------------------------------------------------
    bool TextureRectangle::isIdentifiedBy( const std::string& identifier ) const
    {
        return _memory->isIdentifiedBy( identifier );
    }

	//------------------------------------------------------------------------------
	osgCompute::IdentifierSet& TextureRectangle::getIdentifiers()
	{
		return _memory->getIdentifiers();
	}

	//------------------------------------------------------------------------------
	const osgCompute::IdentifierSet& TextureRectangle::getIdentifiers() const
	{
		return _memory->getIdentifiers();
	}

    //------------------------------------------------------------------------------
    void TextureRectangle::releaseGLObjects( osg::State* state/*=0*/ ) const
    {
        // Currently we support  a single OpenGL context only. So clear memory every
        // time releaseGLObjects() is called.
        //if( state != NULL && 
        //    osgCompute::GLMemory::getContext() != NULL && 
        //    state->getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->clear();

        osg::TextureRectangle::releaseGLObjects( state );
    }

    ////------------------------------------------------------------------------------
    //void TextureRectangle::resizeGLObjectBuffers( unsigned int maxSize )
    //{
    //    // Currently we support  a single OpenGL context only. So clear memory every
    //    // time releaseGLObjects() is called.
    //    //if( osgCompute::GLMemory::getContext() != NULL )
    //    _memory->clear();

    //    osg::Texture::resizeGLObjectBuffers( maxSize );
    //}

    //------------------------------------------------------------------------------
    void TextureRectangle::compileGLObjects(osg::State& state) const
    {
        osg::TextureRectangle::apply(state);
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::apply(osg::State& state) const
    {
        // Currently we support  a single OpenGL context only. So unmap memory every
        // time releaseGLObjects() is called.
        //if( osgCompute::GLMemory::getContext() != NULL && 
        //    state.getContextID() == osgCompute::GLMemory::getContext()->getState()->getContextID() )
        _memory->unmap();

        osg::TextureRectangle::apply( state );
    }

    //------------------------------------------------------------------------------
    void TextureRectangle::applyAsRenderTarget() const
    {
        static_cast<osgCompute::GLMemory*>( _memory )->mapAsRenderTarget();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    TextureRectangle::~TextureRectangle()
    {
		// _memory object is not deleted until this point
		// as reference count is increased in constructor
        _memory->releaseObjects();
		_memory = NULL;
    }
}
