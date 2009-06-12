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

#include <osg/ArgumentParser>
#include <osg/Texture2D>
#include <osg/Viewport>
#include <osg/BlendFunc>
#include <osg/PolygonMode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/PointSprite>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osgViewer/ViewerEventHandlers>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/Registry>
#include <osgCuda/Computation>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>

#include <iostream>
#include <sstream>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

#include "PtclMover"
#include "PtclEmitter"
#include "PtclMapper"

float frand( float minf, float maxf )
{
    float unit = float(rand()) / RAND_MAX;
    float diff = maxf - minf;
    return minf + unit * diff;
}

osg::Geode* getBBox( osg::Vec3& bbmin, osg::Vec3& bbmax )
{
    osg::Geometry* bbgeom = new osg::Geometry;

    // vertices
    osg::Vec3Array* vertices = new osg::Vec3Array();

    osg::Vec3 center = (bbmin + bbmax) * 0.5f;
    osg::Vec3 radiusX( bbmax.x() - center.x(), 0, 0 );
    osg::Vec3 radiusY( 0, bbmax.y() - center.y(), 0 );
    osg::Vec3 radiusZ( 0, 0, bbmax.z() - center.z() );

    vertices->push_back( center - radiusX - radiusY - radiusZ ); // 0
    vertices->push_back( center + radiusX - radiusY - radiusZ ); // 1
    vertices->push_back( center + radiusX + radiusY - radiusZ ); // 2
    vertices->push_back( center - radiusX + radiusY - radiusZ ); // 3
    vertices->push_back( center - radiusX - radiusY + radiusZ ); // 4
    vertices->push_back( center + radiusX - radiusY + radiusZ ); // 5
    vertices->push_back( center + radiusX + radiusY + radiusZ ); // 6
    vertices->push_back( center - radiusX + radiusY + radiusZ ); // 7
    bbgeom->setVertexArray( vertices );

    // indices
    osg::DrawElementsUShort* indices = new osg::DrawElementsUShort(GL_LINES);

    // front
    indices->push_back(0);
    indices->push_back(1);
    indices->push_back(1);
    indices->push_back(2);
    indices->push_back(2);
    indices->push_back(3);
    indices->push_back(3);
    indices->push_back(0);

    // back
    indices->push_back(4);
    indices->push_back(5);
    indices->push_back(5);
    indices->push_back(6);
    indices->push_back(6);
    indices->push_back(7);
    indices->push_back(7);
    indices->push_back(4);

    // rest of line segments
    indices->push_back(1);
    indices->push_back(5);
    indices->push_back(2);
    indices->push_back(6);
    indices->push_back(3);
    indices->push_back(7);
    indices->push_back(0);
    indices->push_back(4);
    bbgeom->addPrimitiveSet( indices );

    // do color
    osg::Vec4Array* color = new osg::Vec4Array;
    color->push_back( osg::Vec4(0.5f, 0.5f, 0.5f, 1.f) );
    bbgeom->setColorArray( color );
    bbgeom->setColorBinding( osg::Geometry::BIND_OVERALL );

    ////////////////
    // SETUP BBOX //
    ////////////////
    osg::Geode* bbox = new osg::Geode;
    bbox->addDrawable( bbgeom );

    // disable lighting
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

    return bbox;
}

osgCuda::Computation* getComputation( unsigned int numParticles, osg::Vec3& bbmin, osg::Vec3& bbmax )
{
    ////////////
    // BUFFER //
    ////////////
    // particle buffer
    osgCuda::Vec4fBuffer* ptclBuffer = new osgCuda::Vec4fBuffer;
    ptclBuffer->setName( "ptclCUDABuffer" );
    ptclBuffer->setDimension(0,numParticles);
    ptclBuffer->addHandle( "PTCL_BUFFER" );
    ptclBuffer->addHandle( "PTCL_CUDA_BUFFER" );

    /////////////
    // MODULES //
    /////////////
    // create module
    PtclDemo::PtclMover* ptclMover = new PtclDemo::PtclMover;
    ptclMover->setName( "ptclMover" );

    PtclDemo::PtclEmitter* ptclEmitter = new PtclDemo::PtclEmitter;
    ptclEmitter->setName( "ptclEmitter" );
    ptclEmitter->setSeedBox( bbmin, bbmax );

    PtclDemo::PtclMapper* ptclMapper = new PtclDemo::PtclMapper;
    ptclMapper->setSyncType( PtclDemo::PtclMapper::SYNC_GL_WITH_CUDA_DEVICE );
    ptclMapper->setName( "ptclMapper" );

    /////////////////
    // COMPUTATION //
    /////////////////
    osgCuda::Computation* computation = new osgCuda::Computation;
    computation->setName("computation");
    computation->addModule( *ptclEmitter );    
    computation->addModule( *ptclMover );
    computation->addModule( *ptclMapper );
    computation->addResource( *ptclBuffer );

    return computation;
}

osg::Geode* getGeode( unsigned int numParticles )
{
    osg::Geode* geode = new osg::Geode;

    // GEOMETRY
    osg::ref_ptr<osgCuda::Vec4fGeometry> ptclGeom = new osgCuda::Vec4fGeometry;

    // initialize the vertices
    osg::Vec4Array* coords = new osg::Vec4Array(numParticles);
    coords->setName("particles");
    for( unsigned int v=0; v<coords->size(); ++v )
        (*coords)[v].set(0,0,0,1);

    //ptclGeom->setUseVertexBufferObjects( true );
    ptclGeom->setVertexArray(coords);
    ptclGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));
    ptclGeom->addHandle( "PTCL_GL_BUFFER" );

    //////////
    // GEOM //
    //////////
    // add particles
    geode->addDrawable( ptclGeom.get() );

    ////////////
    // SPRITE //
    ////////////
    // increase point size within shader
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);

    // use point sprites with alpha blending
    osg::BlendFunc* blend = new osg::BlendFunc;
    geode->getOrCreateStateSet()->setAttribute(blend);
    geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON );

    osg::PointSprite* sprite = new osg::PointSprite();
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);

    // The texture for the sprites
    osg::Texture2D* tex = new osg::Texture2D();
    tex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
    tex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
    tex->setImage(osgDB::readImageFile("osgParticleDemo/images/particle.rgb"));

    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->addUniform(new osg::Uniform("texture",0));

    ////////////
    // SHADER //
    ////////////
    // add program
    osg::Program* program = new osg::Program;
    program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile("osgParticleDemo/shader/PtclSprite.vsh")));
    program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile("osgParticleDemo/shader/PtclSprite.fsh")));
    geode->getOrCreateStateSet()->setAttribute(program);

    // screen resolution for particle sprite
    osg::Uniform* pixelsize = new osg::Uniform();
    pixelsize->setName( "pixelsize" );
    pixelsize->setType( osg::Uniform::FLOAT_VEC2 );
    pixelsize->set( osg::Vec2(1.0f,40.0f) );
    geode->getOrCreateStateSet()->addUniform( pixelsize );
    geode->setCullingActive( false );

    return geode;
}

int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

    osg::Vec3 bbmin(0,0,0);
    osg::Vec3 bbmax(4,4,4);
    unsigned int numParticles = 128000;

    /////////////////
    // SETUP SCENE //
    /////////////////
    osg::Group* scene = new osg::Group;

    osgCompute::Computation* computation = getComputation( numParticles, bbmin, bbmax );
    computation->addChild( getGeode( numParticles ) );
    scene->addChild( computation );
    scene->addChild( getBBox( bbmin, bbmax ) );

    ////////////
    // VIEWER //
    ////////////
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);
    viewer.getCamera()->setComputeNearFarMode( osg::Camera::DO_NOT_COMPUTE_NEAR_FAR );
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );
    viewer.setUpViewInWindow( 50, 50, 640, 480);

    // if you have OSG Version 2.8.1 or the current OSG SVN Version (2.9.1 or later)
    // then try multithreaded
    // otherwise the application will finish with segmentation fault
    viewer.setThreadingModel(osgViewer::Viewer::CullDrawThreadPerContext);
    //viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);

    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);

    return viewer.run();
}
