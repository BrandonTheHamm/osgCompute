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
#include <cuda_runtime.h>
#include <osg/ArgumentParser>
#include <osg/Texture2D>
#include <osg/Vec4ub>
#include <osg/BlendFunc>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Program>
#include <osgCuda/Buffer>
#include <osgCuda/Texture>
#include <osgCudaStats/Stats>

//------------------------------------------------------------------------------
extern "C" void swap( 
                     unsigned int imageWidth,
                     unsigned int imageHeight,
                     void* srcArray, 
                     void* trgBuffer, 
                     unsigned int trgPitch );

class TexFilter : public osgCompute::Computation 
{
public:
    virtual void launch()
    {
        if( !_trgBuffer || !_srcArray )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "TexFilter");
        }

        _timer->start();

        swap(  
            _trgBuffer->getDimension(0),
            _trgBuffer->getDimension(1),
            _srcArray->map( osgCompute::MAP_DEVICE_ARRAY ),
            _trgBuffer->map( osgCompute::MAP_DEVICE_TARGET ),
            _trgBuffer->getPitch() );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("TRG_BUFFER") )
            _trgBuffer = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy("SRC_ARRAY") )
            _srcArray = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    osg::ref_ptr<osgCuda::Timer>     _timer;
    osg::ref_ptr<osgCompute::Memory> _srcArray;
    osg::ref_ptr<osgCompute::Memory> _trgBuffer;
};

//------------------------------------------------------------------------------
osg::ref_ptr<osg::Node> setupScene()
{
    osg::ref_ptr<osg::Group> scene = new osg::Group;

    ///////////////
    // RESOURCES //
    ///////////////
    osg::ref_ptr<osg::Image> srcImage = osgDB::readImageFile("osgTexDemo/images/logo.png");
    if (!srcImage)
    {
        osg::notify(osg::NOTICE) << "main(): Could not open \"osgTexDemo/images/logo.png\" image." << std::endl;
        return NULL;
    }

    // For arrays you have to provide 
    // a channel desc!
    cudaChannelFormatDesc srcDesc;
    srcDesc.f = cudaChannelFormatKindUnsigned;
    srcDesc.x = 8;
    srcDesc.y = 8;
    srcDesc.z = 8;
    srcDesc.w = 8;

    osg::ref_ptr<osgCuda::Buffer> srcArray = new osgCuda::Buffer;
    srcArray->setName( "Source Texture" );
    srcArray->addIdentifier( "SRC_ARRAY" );
    srcArray->setElementSize( sizeof(osg::Vec4ub) );
    srcArray->setChannelFormatDesc( srcDesc );
    srcArray->setDimension( 0, srcImage->s() );
    srcArray->setDimension( 1, srcImage->t() );
    srcArray->setImage( srcImage );

    osg::ref_ptr< osgCuda::Texture2D > trgTexture = new osgCuda::Texture2D;  
    trgTexture->setName( "Target Texture" );
    trgTexture->addIdentifier( "TRG_BUFFER" );
    // Note: GL_RGBA8 Bit format is not yet supported by CUDA, use GL_RGBA8UI_EXT instead.
    // GL_RGBA8UI_EXT requires the additional work of scaling the fragment shader
    // output from 0-1 to 0-255. 	
    trgTexture->setInternalFormat( GL_RGBA32F_ARB );
    trgTexture->setSourceFormat( GL_RGBA );
    trgTexture->setSourceType( GL_FLOAT );
    // in case you choose a texture size which is not a multiple of the alignment restriction CUDA 
    // will allocate more memory to fulfill the alignemnt requirements. To get the
    // actual pitch (bytes of a single row) call osgCompute::Memory::getPitch()
    trgTexture->setTextureWidth( srcImage->s());
    trgTexture->setTextureHeight( srcImage->t() );

    ///////////////////////
    // COMPUTATION SETUP //
    ///////////////////////
    osg::ref_ptr<TexFilter> texFilter = new TexFilter;

    // Execute the program during the rendering, but before
    // the subgraph is rendered. Default is the execution during
    // the update traversal.
    osg::ref_ptr<osgCompute::Program> program = new osgCuda::Program;
    program->setComputeOrder(  osgCompute::Program::PRE_RENDER );
    program->addComputation( *texFilter );
    program->addResource( *srcArray );
    program->addResource( *trgTexture->getMemory() );
    scene->addChild( program );

    /////////////////
    // RESULT QUAD //
    /////////////////
    osg::Geode* quad = new osg::Geode;
    quad->addDrawable( osg::createTexturedQuadGeometry( osg::Vec3(-0.5,0,-0.5), osg::Vec3(1,0,0), osg::Vec3(0,0,1) ) );
    quad->getOrCreateStateSet()->setTextureAttributeAndModes( 0, trgTexture, osg::StateAttribute::ON );
    scene->addChild( quad );

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

    ///////////////
    // RUN SCENE //
    ///////////////
    return viewer.run();
}
