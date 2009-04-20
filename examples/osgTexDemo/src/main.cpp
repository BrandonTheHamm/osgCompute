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
#include <osg/BlendFunc>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <osgCuda/Computation>
#include <osgCuda/IntOpBuffer>
#include <osgCuda/Constant>

#include "TexStreamer"

osgCuda::Computation* getComputation( osg::Image& srcImage, osg::Texture2D& outTexture )
{
    cudaChannelFormatDesc srcDesc;
    srcDesc.f = cudaChannelFormatKindUnsigned;
    srcDesc.x = 8;
    srcDesc.y = 8;
    srcDesc.z = 8;
    srcDesc.w = 8;

    osgCuda::Vec4ubArray* srcArray = new osgCuda::Vec4ubArray;
    srcArray->setName("srcArray");
    srcArray->setChannelFormatDesc( srcDesc );
    srcArray->setDimension( 0, srcImage.s() );
    srcArray->setDimension( 1, srcImage.t() );
    srcArray->setImage( &srcImage, 0 );

    osgCuda::Vec4ubIntOpBuffer* trgBuffer = new osgCuda::Vec4ubIntOpBuffer;
    trgBuffer->setName( "trgBuffer" );
    trgBuffer->setNumStreams( 2 );
    trgBuffer->setIntOpObject( outTexture, 1 );
    trgBuffer->setDimension( 0, srcImage.s() );
    trgBuffer->setDimension( 1, srcImage.t() );

    // Value is used to scale pixel colors
    osg::Vec4f scaleValue( 0.5,0.5,0.5,0.8 );
    osgCuda::Vec4fConstant* scale = new osgCuda::Vec4fConstant;
    scale->setName( "scale" );
    scale->setData( scaleValue );

    TexDemo::TexStreamer* texStreamer = new TexDemo::TexStreamer;
    texStreamer->setName( "MyGoodOldTexStreamer" );

    ///////////
    // SETUP //
    ///////////
    osgCuda::Computation* computation = new osgCuda::Computation;
    computation->addModule( *texStreamer );
    computation->addParamHandle( "TRG_BUFFER", *trgBuffer );
    computation->addParamHandle( "SRC_ARRAY", *srcArray );
    computation->addParamHandle( "SCALE", *scale );

    return computation;
}

osg::Geode* getGeode( osg::Texture2D& trgTexture )
{
    osg::Geode* geode = new osg::Geode;

    osg::Vec3 llCorner = osg::Vec3(-0.5,0,-0.5);
    osg::Vec3 width = osg::Vec3(1,0,0);
    osg::Vec3 height = osg::Vec3(0,0,1);

    //////////
    // QUAD //
    //////////
    osg::Geometry* geom = new osg::Geometry;

    osg::Vec3Array* coords = new osg::Vec3Array(4);
    (*coords)[0] = llCorner;
    (*coords)[1] = llCorner+width;
    (*coords)[2] = llCorner+width+height;
    (*coords)[3] = llCorner+height;

    geom->setVertexArray(coords);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));
    geom->computeBound();
    geode->addDrawable( geom );

    /////////////
    // TEXTURE //
    /////////////
    geode->getOrCreateStateSet()->setTextureAttribute( 0, &trgTexture, osg::StateAttribute::ON );
    osg::Uniform* trgTextureSampler = new osg::Uniform("trgTexture",0);
    geode->getOrCreateStateSet()->addUniform(trgTextureSampler);

    ////////////
    // SHADER //
    ////////////
    osg::Program* program = new osg::Program;
    program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile("osgTexDemo/shader/S0_0.vsh")));
    program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile("osgTexDemo/shader/S0_0.fsh")));
    geode->getOrCreateStateSet()->setAttribute(program);

    ///////////////
    // STATESETS //
    ///////////////
    osg::BlendFunc* blend = new osg::BlendFunc;
    geode->getOrCreateStateSet()->setAttribute(blend);
    geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON );

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

    // TARGET
    osg::ref_ptr<osg::Texture2D> trgTexture = new osg::Texture2D;
    trgTexture->setName("myTarget");
    trgTexture->setTextureWidth(srcImage->s());
    trgTexture->setTextureHeight(srcImage->t());
    trgTexture->setInternalFormat( GL_RGBA );
    trgTexture->setSourceType( GL_UNSIGNED_BYTE );
    trgTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    trgTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);

    ///////////
    // SCENE //
    ///////////
    osg::Group* scene = new osg::Group;
    // COMPUTATION
    scene->addChild( getComputation( *srcImage, *trgTexture ) );
    // QUAD
    scene->addChild( getGeode( *trgTexture ) );

    ////////////
    // VIEWER //
    ////////////
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);
    viewer.setUpViewInWindow( 50, 50, 640, 480);

    // if you have OSG Version 2.8.1 or the current OSG SVN Version (2.9.1 or later)
    // then try to run it multi-threaded
    // otherwise the application will finish with segmentation fault
    viewer.setThreadingModel(osgViewer::Viewer::DrawThreadPerContext);
    //viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);

    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);

    return viewer.run();
}
