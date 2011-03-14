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
#include <osgCuda/Computation>
#include <osgCuda/Buffer>
#include <osgCuda/Texture>

#include "TexFilter"

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
osg::ref_ptr<osgCompute::Computation> setupComputation()
{
    osg::ref_ptr<osgCompute::Computation> computationNode = new osgCuda::Computation;

    ///////////////
    // RESOURCEN //
    ///////////////
    // SOURCE
    osg::ref_ptr<osg::Image> srcImage = osgDB::readImageFile("osgTexDemo/images/logo.png");
    if (!srcImage)
    {
        osg::notify(osg::NOTICE) << "main(): Could not open \"osgTexDemo/images/logo.png\" image." << std::endl;
        return NULL;
    }

    // for arrays you have to provide 
    // a channel desc!!!!
    cudaChannelFormatDesc srcDesc;
    srcDesc.f = cudaChannelFormatKindUnsigned;
    srcDesc.x = 8;
    srcDesc.y = 8;
    srcDesc.z = 8;
    srcDesc.w = 8;

    osg::ref_ptr<osgCuda::Buffer> srcArray = new osgCuda::Buffer;
    srcArray->setElementSize( sizeof(osg::Vec4ub) );
    srcArray->setChannelFormatDesc( srcDesc );
    srcArray->setDimension( 0, srcImage->s() );
    srcArray->setDimension( 1, srcImage->t() );
    srcArray->setImage( srcImage );
    // Mark this buffer as a sry array
    srcArray->addIdentifier( "SRC_ARRAY" );

    osg::ref_ptr< osgCuda::Texture2D > trgTexture = new osgCuda::Texture2D;  
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
    trgTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    trgTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    // Mark this buffer as the target buffer of the module
    trgTexture->addIdentifier( "TRG_BUFFER" );

    //////////////////
    // MODULE SETUP //
    //////////////////
    osg::ref_ptr<TexDemo::TexFilter> texFilter = new TexDemo::TexFilter;

    // Execute the computation during the rendering, but before
    // the subgraph is rendered. Default is the execution during
    // the update traversal.
    computationNode->setComputeOrder(  osgCompute::Computation::PRERENDER_BEFORECHILDREN );
    computationNode->addModule( *texFilter );
    computationNode->addResource( *srcArray );
    computationNode->addResource( *trgTexture->getOrCreateInteropMemory() );
    // the target texture is located in the subgraph of the computation
    computationNode->addChild( getTexturedQuad( *trgTexture ) );

    // Write this computation to file
    //osgDB::writeNodeFile( *computationNode, "texdemo.osgt" );

    return computationNode;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);

    /////////////////
    // COMPUTATION //
    /////////////////
    osg::ref_ptr<osgCompute::Computation> computation = setupComputation();

    //////////////////
    // VIEWER SETUP //
    //////////////////
    osg::Group* scene = new osg::Group;
    scene->addChild( computation );

    // You must use single threaded version since osgCompute currently
    // does only support single threaded applications. Please ask in the
    // forum for the multi-threaded version if you need it.
    viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);
    viewer.setReleaseContextAtEndOfFrameHint(false);
    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );

    return viewer.run();
}
