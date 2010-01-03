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
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Computation>
#include <osgCuda/Array>
#include <osgCuda/Texture>

#include "TexStreamer"

osg::Geode* getTexturedQuad( osg::Texture2D& trgTexture )
{
    osg::Geode* geode = new osg::Geode;
    geode->setName("quad");

    osg::Vec3 llCorner = osg::Vec3(-0.5,0,-0.5);
    osg::Vec3 width = osg::Vec3(1,0,0);
    osg::Vec3 height = osg::Vec3(0,0,1);

    //////////
    // QUAD //
    //////////
    osg::Geometry* geom = osg::createTexturedQuadGeometry( llCorner, width, height );
    geode->addDrawable( geom );
    geode->getOrCreateStateSet()->setTextureAttributeAndModes( 0, &trgTexture, osg::StateAttribute::ON );

    return geode;
}

int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

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

    osgCuda::Array* srcArray = new osgCuda::Array;
    srcArray->setName("srcArray");
    srcArray->setElementSize( sizeof(osg::Vec4ub) );
    srcArray->setChannelFormatDesc( srcDesc );
    srcArray->setDimension( 0, srcImage->s() );
    srcArray->setDimension( 1, srcImage->t() );
    srcArray->setImage( srcImage );
    // mark this buffer as a sry array
    srcArray->addHandle( "SRC_ARRAY" );


    osg::ref_ptr< osgCuda::Texture2D > trgTexture = new osgCuda::Texture2D;
    //trgTexture->setInternalFormat( GL_RGBA8UI_EXT );
    //trgTexture->setSourceFormat( GL_RGBA_INTEGER_EXT );
    //trgTexture->setSourceType( GL_UNSIGNED_BYTE );
    trgTexture->setInternalFormat( GL_RGBA );
    trgTexture->setSourceFormat( GL_RGBA );
    trgTexture->setSourceType( GL_UNSIGNED_BYTE );

    trgTexture->setName( "trgBuffer" );
    trgTexture->setTextureWidth(srcImage->s());
    trgTexture->setTextureHeight(srcImage->t());
    trgTexture->setResizeNonPowerOfTwoHint(false);
    trgTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    trgTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    // mark this buffer as the target buffer which
    // is displayed
    trgTexture->addHandle( "TRG_BUFFER" );

    //////////////////
    // MODULE SETUP //
    //////////////////
    TexDemo::TexStreamer* texStreamer = new TexDemo::TexStreamer;
    texStreamer->setName( "my texture module" );

    osgCuda::Computation* computation = new osgCuda::Computation;
    // execute the computation during the rendering, but before
    // the subgraph is rendered. Default is the execution during
    // the update traversal.
    computation->setComputeOrder(  osgCompute::Computation::RENDER_PRE_RENDER_PRE_TRAVERSAL );
    computation->addModule( *texStreamer );
    computation->addResource( *srcArray );
    // trgTexture is located in the subgraph of the computation
    computation->addChild( getTexturedQuad( *trgTexture ) );

    /////////////////
    // SCENE SETUP //
    /////////////////
    osg::Group* scene = new osg::Group;
    //scene->addChild( getTexturedQuad(*trgTexture) );
    scene->addChild( computation );

    //////////////////
    // VIEWER SETUP //
    //////////////////
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );

    // You must use single threaded version since osgCompute currently
    // does only support single threaded applications. Please ask in the
    // forum for the multi-threaded version if you need it.
    viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);

    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);

    return viewer.run();
}
