/* osgCompute - Copyright (C) 2008-2009 SVT Group
*
* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of
* the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesse General Public License for more details.
*
* The full license is in LICENSE file included with this distribution.
*/

#include <iostream>
#include <sstream>
#include <osg/ArgumentParser>
#include <osg/Texture2D>
#include <osg/Vec4ub>
#include <osg/BlendFunc>
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Program>
#include <osgCompute/Callback>
#include <osgCuda/Memory>
#include <osgCuda/Texture>
#include <osgCudaStats/Stats>

#include <cuda_runtime.h>

//-------------------------------------------------------------------------
// The filter is implemented in TexFilter.cu
extern "C" void sobelFilter( 
                   const dim3& blocks, 
                   const dim3& threads, 
                   void* trgBuffer, 
                   void* srcBuffer, 
                   unsigned int srcBufferSize );

/** This texture filter launches a sobel filter on the rendered texture of a camera.
    It is executed directly after the camera during the rendering traversal. The 
    result of this computation is written into a target texture which is rendered 
    via a screen aligned quad.
*/
class TexFilter : public osgCompute::Computation 
{
public:
    //------------------------------------------------------------------------------
    virtual void launch()
    {
        if( !_trgBuffer.valid() || !_srcBuffer.valid() )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "TexFilter");
            _timer->init();
        }

        _timer->start();

        dim3 threads = dim3( 16, 16, 1 );
        dim3 blocks = dim3( 
            _trgBuffer->getDimension(0)/16, 
            _trgBuffer->getDimension(1)/16, 1 );

        sobelFilter(  blocks, 
                      threads,
                      _trgBuffer->map( osgCompute::MAP_DEVICE_TARGET ),
                      _srcBuffer->map( osgCompute::MAP_DEVICE_SOURCE ),
                      _srcBuffer->getByteSize( osgCompute::MAP_DEVICE ) );


        _timer->stop();
    }

    //------------------------------------------------------------------------------
    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy( "TRG_BUFFER" ) )
            _trgBuffer = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy( "SRC_BUFFER" ) )
            _srcBuffer = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    osg::ref_ptr<osgCompute::Memory> _srcBuffer;
    osg::ref_ptr<osgCompute::Memory> _trgBuffer;
    osg::ref_ptr<osgCuda::Timer>     _timer;
};


//------------------------------------------------------------------------------
osg::ref_ptr<osg::Node> setupScene()
{
    /////////////////////////
    // LOAD SCENE GEOMETRY //
    /////////////////////////
    osg::ref_ptr<osg::MatrixTransform> loadedModelTransform = new osg::MatrixTransform;
    { 
        osg::ref_ptr<osg::Node> loadedModel = osgDB::readNodeFile("cow.osg");
        if (!loadedModel)
        {
            // If not loaded try use default model instead.
            osg::TessellationHints* hints = new osg::TessellationHints;
            hints->setDetailRatio(0.5f);
            
            osg::ref_ptr<osg::Geode> geode = new osg::Geode;
            geode->addDrawable( new osg::ShapeDrawable(new osg::Cone(osg::Vec3(4.0f,0.0f,0.0f),0.8f,1.0f),hints) );
            loadedModel = geode;
        }

        // Create a transform to spin the model.
        loadedModelTransform->addChild(loadedModel);

        // Rotate the model automatically 
        loadedModelTransform->setUpdateCallback( new osg::AnimationPathCallback(
            loadedModelTransform->getBound().center(),osg::Vec3(0.0f,0.0f,1.0f),osg::inDegrees(45.0f)) );
    }

    //////////////////////////////
    // CREATE TEXTURE RESOURCES //
    //////////////////////////////
    osg::ref_ptr< osgCuda::Texture2D > rttTexture = new osgCuda::Texture2D;
    {
        rttTexture->setTextureSize(512, 512);
        rttTexture->setInternalFormat(GL_RGBA);
        rttTexture->setSourceType( GL_UNSIGNED_BYTE );
        rttTexture->setName( "srcBuffer" );
        rttTexture->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR);
        rttTexture->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);
        rttTexture->addIdentifier( "SRC_BUFFER" );
    }

    osg::ref_ptr< osgCuda::Texture2D > targetTexture = new osgCuda::Texture2D;
    {
        targetTexture->setTextureSize(512, 512);
        targetTexture->setInternalFormat(GL_RGBA);
        targetTexture->setSourceType( GL_UNSIGNED_BYTE );
        targetTexture->setName( "trgBuffer" );
        targetTexture->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR);
        targetTexture->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);  
        targetTexture->addIdentifier( "TRG_BUFFER" );
    }

    ///////////////////////////////////
    // SETUP RENDER TO TEXTURE GRAPH //
    ///////////////////////////////////
    // Create the camera node to do the render to texture
    osg::ref_ptr<osg::Camera> rttCamera = new osg::Camera;
    {    
        rttCamera->setName( "PRERENDER_TO_TEXTURE_CAMERA" );

        // set up the background color and clear mask.
        rttCamera->setClearColor(osg::Vec4(0,0,0,1.0f));
        rttCamera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const osg::BoundingSphere& bs = loadedModelTransform->getBound();     

        float znear = 1.0f*bs.radius();
        float zfar  = 3.0f*bs.radius();

        // 2:1 aspect ratio as per flag geometry below.
        float proj_top   = 0.25f*znear;
        float proj_right = 0.5f*znear;

        znear *= 0.9f;
        zfar *= 1.1f;

        // set up projection.
        rttCamera->setProjectionMatrixAsFrustum(-proj_right,proj_right,-proj_top,proj_top,znear,zfar);

        // set view
        rttCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        rttCamera->setViewMatrixAsLookAt(bs.center()-osg::Vec3(0.0f,2.0f,0.0f)*bs.radius(),bs.center(),osg::Vec3(0.0f,0.0f,1.0f));

        // set viewport
        rttCamera->setViewport(0,0,rttTexture->getTextureWidth(),rttTexture->getTextureHeight());

        // set the camera to render before the main camera.
        rttCamera->setRenderOrder(osg::Camera::PRE_RENDER);

        // tell the camera to use OpenGL frame buffer object where supported.
        rttCamera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);

        // attach the texture and use it as the color buffer.
        rttCamera->attach(osg::Camera::COLOR_BUFFER, rttTexture, 0, 0, false, 0, 0 );

        // add subgraph to render
        rttCamera->addChild(loadedModelTransform);

        // Add a GLMemoryTargetCallback to the camera in order to inform the render target 
        // when the camera renders into it.
        osg::ref_ptr<osgCompute::GLMemoryTargetCallback> rttCallback = new osgCompute::GLMemoryTargetCallback;
        rttCallback->observe( rttTexture->getMemory() );
        rttCamera->setPreDrawCallback( rttCallback );
    }  

    //////////////////////////////
    // SETUP OSGCOMPUTE PROGRAM //
    //////////////////////////////
    // Create a program. Please note that you do not need
    // to execute your CUDA kernels in programs only. However,
    // in order to get the OpenGL rendering correctly scheduled 
    // with computations, programs are useful.
    osg::ref_ptr<osgCuda::Program> program = new osgCuda::Program;
    {
        // Add the filter computation to the program and ...
        osg::ref_ptr<TexFilter> texFilter = new TexFilter;
        program->addComputation( *texFilter );
        // ... execute it during the render traversal. Programs
        // are handled just like osg::Camera objects during
        // the rendering-traversal.
        program->setComputeOrder( osgCompute::Program::PRE_RENDER ); 

        // Add resources manually. Please see osgCompute::ResourceVisitor
        // for an automatic setup of resources.
        program->addResource( *(rttTexture->getMemory()) );
        program->addResource( *(targetTexture->getMemory()) );

        // Add the camera as a pre-render subgraph
        program->addChild( rttCamera );
    }

    ////////////////////////
    // ILLUSTRATIVE SCENE //
    ////////////////////////
    // Translate the original model - just for illustration purpose
    osg::ref_ptr<osg::MatrixTransform> sceneTransform = new osg::MatrixTransform;
    {
        osg::Matrix m;
        m.makeScale( 0.2, 0.2, 0.2 );
        m.setTrans(-2.0, 0.0, 0.0 );
        sceneTransform->setMatrix( m );
        sceneTransform->addChild( loadedModelTransform );
    }

    /////////////////
    // RESULT QUAD //
    /////////////////
    osg::ref_ptr<osg::Geode> resultQuad = new osg::Geode;
    {
        resultQuad->setDataVariance( osg::Object::DYNAMIC );
        resultQuad->addDrawable( osg::createTexturedQuadGeometry( osg::Vec3(-0.5,0,-0.5), osg::Vec3(1,0,0), osg::Vec3(0,0,1) ) );
        resultQuad->getOrCreateStateSet()->setTextureAttributeAndModes( 0, targetTexture.get(), osg::StateAttribute::ON );
    }

    ////////////////
    // TEST SCENE //
    ////////////////
    osg::ref_ptr<osg::Group> scene = new osg::Group;
    {
        scene->addChild( sceneTransform );
        scene->addChild( resultQuad );
        scene->addChild( program );
    }

    return scene;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

    //////////////////
    // SETUP VIEWER //
    //////////////////
    osgViewer::Viewer viewer;
    viewer.addEventHandler(new osgViewer::StatsHandler);
    viewer.addEventHandler(new osgCuda::StatsHandler);
    viewer.addEventHandler(new osgViewer::HelpHandler);
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );

    ///////////////////////
    // LINK CUDA AND OSG //
    ///////////////////////
    // setupOsgCudaAndViewer() creates
    // the OpenGL context and binds
    // it to the CUDA context of the thread.
    osgCuda::setupOsgCudaAndViewer( viewer );

    /////////////////
    // SETUP SCENE //
    /////////////////
    viewer.setSceneData( setupScene() );

    //////////
    // LOOP //
    //////////
    return viewer.run();
}
