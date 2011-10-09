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
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Program>
#include <osgCuda/Memory>
#include <osgCuda/Texture>

#include "TexFilter"

//------------------------------------------------------------------------------
osg::Geode* getTexturedQuad( osg::Texture2D& trgTexture )
{
    osg::Geode* geode = new osg::Geode;
    osg::Vec3 llCorner = osg::Vec3(-0.5,0,-0.5);
    osg::Vec3 width = osg::Vec3(1,0,0);
    osg::Vec3 height = osg::Vec3(0,0,1);

    //////////
    // QUAD //
    //////////
    osg::ref_ptr<osg::Geometry> geom = osg::createTexturedQuadGeometry( llCorner, width, height );
    geode->addDrawable( geom );
    geode->getOrCreateStateSet()->setTextureAttributeAndModes( 0, &trgTexture, osg::StateAttribute::ON );
    geode->getOrCreateStateSet()->addUniform( 
        new osg::Uniform( "texImage", 0 ) );

    return geode;
}

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::Program> setupProgram()
{
    osg::ref_ptr<osgCompute::Program> program = new osgCuda::Program;

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
    // a channel desc!!!!
    cudaChannelFormatDesc srcDesc;
    srcDesc.f = cudaChannelFormatKindUnsigned;
    srcDesc.x = 8;
    srcDesc.y = 8;
    srcDesc.z = 8;
    srcDesc.w = 8;

    osg::ref_ptr<osgCuda::Memory> srcArray = new osgCuda::Memory;
    srcArray->setElementSize( sizeof(osg::Vec4ub) );
    srcArray->setChannelFormatDesc( srcDesc );
    srcArray->setDimension( 0, srcImage->s() );
    srcArray->setDimension( 1, srcImage->t() );
    srcArray->setImage( srcImage );
    // Mark this buffer as the source array of the module
    srcArray->addIdentifier( "SRC_ARRAY" );

    osg::ref_ptr< osgCuda::Texture2D > trgTexture = new osgCuda::Texture2D;  
	trgTexture->setName("My Target Texture");
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
    // Mark this buffer as the target buffer of the module
    trgTexture->addIdentifier( "TRG_BUFFER" );

    //////////////////
    // MODULE SETUP //
    //////////////////
    osg::ref_ptr<TexDemo::TexFilter> texFilter = new TexDemo::TexFilter;

    // Execute the program during the rendering, but before
    // the subgraph is rendered. Default is the execution during
    // the update traversal.
    program->setComputeOrder(  osgCompute::Program::PRERENDER_BEFORECHILDREN );
    program->addComputation( *texFilter );
    program->addResource( *srcArray );
    program->addResource( *trgTexture->getMemory() );
    // the target texture is located in the subgraph of the program
    program->addChild( getTexturedQuad( *trgTexture ) );

    // Serialize the program to file by activating the
    // following line of code:
    // osgDB::writeNodeFile( *program, "texdemo.osgt" );
    // Afterwards you can load it via:
    // osg::ref_ptr<osg::Node> program = osgDB::readNodeFile( "PATH_TO_FILE/texdemo.osgt" );

    return program;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

    osgViewer::Viewer viewer( osg::ArgumentParser(&argc, argv) );
    viewer.addEventHandler(new osgViewer::StatsHandler);
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );

    //////////////////
    // SETUP VIEWER //
    //////////////////
    osgCuda::setupOsgCudaAndViewer( viewer );

    /////////////////
    // SETUP SCENE //
    /////////////////
    osg::ref_ptr<osgCompute::Program> program = setupProgram();
    viewer.setSceneData( program );

    return viewer.run();
}
