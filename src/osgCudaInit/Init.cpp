#include <osg/GL>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_gl_interop.h>
#include <osg/OperationThread>
#include <osgCompute/Memory>
#include <osgCudaInit/Init>

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
            osg::notify(osg::FATAL)  
                << __FUNCTION__ << ": error during cudaGetDeviceCount()."
                << cudaGetErrorString(res)
                << std::endl;

            return false;
        }

        if( device > deviceCount - 1 )
        {
            osg::notify(osg::FATAL)   
                << __FUNCTION__ << ": device \""<<device<<"\" does not exist."
                << std::endl;

            return false;
        }

        res = cudaSetDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << __FUNCTION__ << ": cannot setup device."
                << cudaGetErrorString(res) 
                << std::endl;
            return false;
        }

        res = cudaGLSetGLDevice( device );
        if( cudaSuccess != res )
        {
            osg::notify(osg::FATAL)  
                << __FUNCTION__ << ": cannot share device with OpenGL."
                << cudaGetErrorString(res) 
                << std::endl;
            return false;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool setupDeviceAndContext( osg::GraphicsContext& ctx, int device /*= 0*/ )
    {
        if( ctx.getState() == NULL )
        {
            osg::notify(osg::WARN)   
                << __FUNCTION__ << ": osg::GraphicsContext must have a valid state."
                << std::endl;

            return false;
        }

        // Use first context to be found and make it current.
        ctx.makeCurrent();

        if( NULL != osgCompute::GLMemory::getContext() && 
            osgCompute::GLMemory::getContext()->getState()->getContextID() != ctx.getState()->getContextID() )
        {
            osg::notify(osg::WARN)   
                << __FUNCTION__ << ": osgCuda can handle only a single context."
                << " However multiple contexts are detected."
                << " Please make sure to share a GL context among all windows."
                << std::endl;

            return false;
        }

        // Bind context to osgCompute::GLMemory
        if( osgCompute::GLMemory::getContext() == NULL )
            osgCompute::GLMemory::bindToContext( ctx );

        return setupDevice(device);
    }

    //------------------------------------------------------------------------------
    // Setup viewer to properly handle osgCuda.
    bool setupOsgCudaAndViewer( osgViewer::ViewerBase& viewer, int ctxID /*= -1*/, int device /*= 0 */ )
    {
        // You must use single threaded version since osgCompute currently
        // does only support single threaded applications. 
        viewer.setThreadingModel( osgViewer::ViewerBase::SingleThreaded );

        // Does create a single OpenGL context
        // which is not released at the end of a frame to secure 
        // CUDA launches everywhere
        viewer.setReleaseContextAtEndOfFrameHint(false);

        // Create the current OpenGL context and make it current
        if( !viewer.isRealized() )
            viewer.realize();

        osgViewer::ViewerBase::Contexts ctxs;
        viewer.getContexts( ctxs, true );
        if( ctxs.empty() )
        {
            osg::notify(osg::FATAL)<< __FUNCTION__ << ": no valid OpenGL context is found."<<std::endl;
            return false;
        }

        osg::GraphicsContext* ctx = NULL;
        if( ctxID != -1 )
        {   // Find context with ctxID and make it current.
            for( unsigned int c=0; c<ctxs.size(); ++c )
            {
                if( ctxs[c]->getState()->getContextID() == ctxID )
                {   
                    ctx = ctxs[c];
                }
            }
        }
        else
        {   
            ctx = ctxs.front();
        }

        if( NULL == ctx )
        {
            osg::notify(osg::FATAL)<< __FUNCTION__ << ": cannot find valid OpenGL context."<<std::endl;
            return false;
        }

        // Connect the CUDA device with OpenGL
        if( !setupDeviceAndContext( *ctx, device ) )
        {
            osg::notify(osg::FATAL)<< __FUNCTION__ << ": cannot setup OpenGL with CUDA."<<std::endl;
            return false;
        }

        return true;
    }
} 
