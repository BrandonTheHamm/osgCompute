#include <osg/gl>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include <osg/NodeVisitor>
#include <osg/OperationThread>
#include <osgUtil/CullVisitor>
#include <osgCuda/Computation>

namespace osgCuda
{ 
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Computation::setupDevice( int device )
    {
        int deviceCount = 0;
        cudaError res = cudaGetDeviceCount( &deviceCount );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Computation::setupDevice(): error during cudaGetDeviceCount()."
                << cudaGetErrorString(res)
                << std::endl;

            return;
        }

        if( device > deviceCount - 1 )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Computation::setupDevice(): device \""<<device<<"\" does not exist."
                << std::endl;

            return;
        }

        res = cudaSetDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "osgCuda::Computation::setupDevice(): cannot setup device."
                << cudaGetErrorString(res) 
                << std::endl;
            return;
        }

        res = cudaGLSetGLDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "osgCuda::Computation::setupDevice(): cannot share device with OpenGL."
                << cudaGetErrorString(res) 
                << std::endl;
            return;
        }

        osgCompute::Computation::setDeviceReady();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Computation::Computation() 
        :   osgCompute::Computation()
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------
    void Computation::clear() 
    { 
        clearLocal();
        osgCompute::Computation::clear(); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Computation::clearLocal() 
    {
    }

    //------------------------------------------------------------------------------
    void osgCuda::Computation::checkDevice()
    {
        if( !osgCompute::Computation::isDeviceReady() ) 
            setupDevice( 0 );
    }
} 
