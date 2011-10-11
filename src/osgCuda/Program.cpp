#include <osg/gl>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include <osg/NodeVisitor>
#include <osg/OperationThread>
#include <osgUtil/CullVisitor>
#include <osgCompute/Memory>
#include <osgCuda/Program>
#include <osg/Camera>

namespace osgCuda
{ 
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool setupDevice( int device )
    {
        int deviceCount = 0;
        cudaError res = cudaGetDeviceCount( &deviceCount );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Program::setupDevice(): error during cudaGetDeviceCount()."
                << cudaGetErrorString(res)
                << std::endl;

            return false;
        }

        if( device > deviceCount - 1 )
        {
            osg::notify(osg::FATAL)  << "osgCuda::Program::setupDevice(): device \""<<device<<"\" does not exist."
                << std::endl;

            return false;
        }

        res = cudaSetDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "osgCuda::Program::setupDevice(): cannot setup device."
                << cudaGetErrorString(res) 
                << std::endl;
            return false;
        }

        res = cudaGLSetGLDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "osgCuda::Program::setupDevice(): cannot share device with OpenGL."
                << cudaGetErrorString(res) 
                << std::endl;
            return false;
        }

        osgCompute::setDeviceReady();
        return true;
    }

    //------------------------------------------------------------------------------
    bool setupOsgCudaAndViewer( osgViewer::ViewerBase& viewer, int device /*= 0 */, bool realize /*= true*/ )
    {
        // Setup viewer to properly handle osgCuda.

        // You must use single threaded version since osgCompute currently
        // does only support single threaded applications. Please ask in the
        // forum for the multi-threaded version if you need it.
        viewer.setThreadingModel( osgViewer::ViewerBase::SingleThreaded );

        // Does create a single OpenGL context
        // which is not released at the end of a frame to secure 
        // CUDA launches everywhere
        viewer.setReleaseContextAtEndOfFrameHint(false);

        if( realize )
        {
            // Create the current OpenGL context and make it current
            viewer.realize();
            osgViewer::ViewerBase::Contexts ctxs;
            viewer.getContexts( ctxs, true );
            if( ctxs.empty() )
            {
                osg::notify(osg::WARN)<<"Cannot setup CUDA context and viewer as no valid OpenGL context is found"<<std::endl;
                return false;
            }
            ctxs.front()->makeCurrent();

            // Bind Context to osgCompute::GLMemory
            osgCompute::GLMemory::bindToContext( *ctxs.front() );
        }


        return setupDevice( device );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Program::Program() 
        :   osgCompute::Program()
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------
    void Program::clear() 
    { 
        clearLocal();
        osgCompute::Program::clear(); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Program::clearLocal() 
    {
    }

    //------------------------------------------------------------------------------
    void osgCuda::Program::checkDevice()
    {
        if( !osgCompute::isDeviceReady() ) 
            setupDevice( 0 );
    }
} 
