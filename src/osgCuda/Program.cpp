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
            osg::notify(osg::FATAL)  << "setupDevice(): error during cudaGetDeviceCount()."
                << cudaGetErrorString(res)
                << std::endl;

            return false;
        }

        if( device > deviceCount - 1 )
        {
            osg::notify(osg::FATAL)  << "setupDevice(): device \""<<device<<"\" does not exist."
                << std::endl;

            return false;
        }

        res = cudaSetDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "setupDevice(): cannot setup device."
                << cudaGetErrorString(res) 
                << std::endl;
            return false;
        }

        res = cudaGLSetGLDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << "setupDevice(): cannot share device with OpenGL."
                << cudaGetErrorString(res) 
                << std::endl;
            return false;
        }

        osgCompute::setDeviceReady();
        return true;
    }

    //------------------------------------------------------------------------------
    // Setup viewer to properly handle osgCuda.
    bool setupOsgCudaAndViewer( osgViewer::ViewerBase& viewer, bool realize /*= true*/, int device /*= 0 */, int ctxID /*= -1*/  )
    {
        // You must use single threaded version since osgCompute currently
        // does only support single threaded applications. 
        viewer.setThreadingModel( osgViewer::ViewerBase::SingleThreaded );

        // Does create a single OpenGL context
        // which is not released at the end of a frame to secure 
        // CUDA launches everywhere
        viewer.setReleaseContextAtEndOfFrameHint(false);

        if( realize )
        {
            // Create the current OpenGL context and make it current
            if( !viewer.isRealized() )
                viewer.realize();

            osgViewer::ViewerBase::Contexts ctxs;
            viewer.getContexts( ctxs, true );
            if( ctxs.empty() )
            {
                osg::notify(osg::FATAL)<<"setupOsgCudaAndViewer(): no valid OpenGL context is found."<<std::endl;
                return false;
            }

            if( ctxID != -1 )
            {   // Find context with ctxID and make it current.
                for( unsigned int c=0; c<ctxs.size(); ++c )
                {
                    if( ctxs[c]->getState()->getContextID() == ctxID )
                    {   // Make the context current
                        ctxs[c]->makeCurrent();
                        // Bind Context to osgCompute::GLMemory
                        osgCompute::GLMemory::bindToContext( *ctxs.front() );
                    }
                }
            }
            else
            {   // Use first context to be found and make it current.
                ctxs.front()->makeCurrent();
                // Bind Context to osgCompute::GLMemory
                osgCompute::GLMemory::bindToContext( *ctxs.front() );
            }

            if( NULL == osgCompute::GLMemory::getContext() )
            {
                osg::notify(osg::FATAL)<<"setupOsgCudaAndViewer(): cannot find valid OpenGL context."<<std::endl;
                return false;
            }

            // Connect the CUDA device with OpenGL
            if( !setupDevice(device) )
            {
                osg::notify(osg::FATAL)<<"setupOsgCudaAndViewer(): cannot setup CUDA context."<<std::endl;
                return false;
            }
        }

        return true;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Program::Program() 
        :   osgCompute::Program()
    { 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void osgCuda::Program::checkDevice()
    {
        if( !osgCompute::isDeviceReady() ) 
            setupDevice( 0 );
    }
} 
