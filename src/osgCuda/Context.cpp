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
#include <osgCuda/Context>

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
                << "osgCuda::Context::init(): cannot share device with render context."
                << std::endl;

            return false;
        }

        s_sharedDevices.insert( device );
        return true;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Context::Context()
        : osgCompute::Context()
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
            osg::notify(osg::FATAL)  << "osgCuda::Context::init(): error during cudaGetDeviceCount(). Returned code is "<<res<<"."
                << std::endl;
            clear();
            return false;
        }

        if( _device > deviceCount - 1 )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Context::init(): device \""<<_device<<"\" does not exist."
                << std::endl;

            clear();
            return false;
        }

        cudaDeviceProp deviceProp;
        res = cudaGetDeviceProperties( &deviceProp, _device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Context::init(): no device found which supports CUDA."
                << std::endl;

            clear();
            return false;
        }

        if( deviceProp.major < 1 )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Context::init(): device does not support CUDA.\n"
                << std::endl;

            clear();
            return false;
        }

        _deviceProperties = deviceProp;


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
                    << "osgCuda::Context::init(): cannot share device with render context."
                    << std::endl;

                clear();
                return false;
            }
        }

        return osgCompute::Context::init();
    }

    //------------------------------------------------------------------------------
    void Context::apply()
    {
        osgCompute::Context::apply();
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
        memset( &_deviceProperties, 0x0, sizeof(cudaDeviceProp) );
    }
}