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
#include <osg/Viewport>
#include <osg/AlphaFunc>
#include <osg/PolygonMode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/Array>
#include <osg/PointSprite>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Computation>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>

#include "PtclMover"
#include "PtclEmitter"


osg::Geode* getBoundingBox( osg::Vec3& bbmin, osg::Vec3& bbmax )
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

    indices->push_back(0);
    indices->push_back(1);
    indices->push_back(1);
    indices->push_back(2);
    indices->push_back(2);
    indices->push_back(3);
    indices->push_back(3);
    indices->push_back(0);

    indices->push_back(4);
    indices->push_back(5);
    indices->push_back(5);
    indices->push_back(6);
    indices->push_back(6);
    indices->push_back(7);
    indices->push_back(7);
    indices->push_back(4);

    indices->push_back(1);
    indices->push_back(5);
    indices->push_back(2);
    indices->push_back(6);
    indices->push_back(3);
    indices->push_back(7);
    indices->push_back(0);
    indices->push_back(4);
    bbgeom->addPrimitiveSet( indices );

    // color
    osg::Vec4Array* color = new osg::Vec4Array;
    color->push_back( osg::Vec4(0.5f, 0.5f, 0.5f, 1.f) );
    bbgeom->setColorArray( color );
    bbgeom->setColorBinding( osg::Geometry::BIND_OVERALL );

    ////////////////
    // SETUP BBOX //
    ////////////////
    osg::Geode* bbox = new osg::Geode;
    bbox->addDrawable( bbgeom );

    // Disable lighting
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

    return bbox;
}

osgCuda::Computation* getComputation( osg::Vec3& bbmin, osg::Vec3& bbmax )
{
    ///////////
    // SEEDS //
    ///////////
    unsigned int seedCount = 64000;
    osg::FloatArray* seedValues = new osg::FloatArray();
    for( unsigned int s=0; s<seedCount; ++s )
        seedValues->push_back( float(rand()) / RAND_MAX );

    osgCuda::Buffer* seedBuffer = new osgCuda::Buffer;
	seedBuffer->setElementSize( sizeof(float) );
    seedBuffer->setName( "ptclSeedBuffer" );
    seedBuffer->setDimension(0,seedCount);
    seedBuffer->setArray( seedValues );
    seedBuffer->addHandle( "PTCL_SEEDS" );

    /////////////
    // MODULES //
    /////////////
    // create module
    PtclDemo::PtclMover* ptclMover = new PtclDemo::PtclMover;
    ptclMover->setName( "ptclMover" );

    PtclDemo::PtclEmitter* ptclEmitter = new PtclDemo::PtclEmitter;
    ptclEmitter->setName( "ptclEmitter" );
    ptclEmitter->setSeedBox( bbmin, bbmax );

    /////////////////
    // COMPUTATION //
    /////////////////
    osgCuda::Computation* computation = new osgCuda::Computation;
    computation->setName("computation");
    computation->addModule( *ptclEmitter );    
    computation->addModule( *ptclMover );
    computation->addResource( *seedBuffer );

    return computation;
}

osg::Geode* getGeode( unsigned int numParticles )
{
    osg::Geode* geode = new osg::Geode;

	//////////////
	// GEOMETRY //
	//////////////
    osg::ref_ptr<osgCuda::Geometry> ptclGeom = new osgCuda::Geometry;

    // Initialize the Particles
    osg::Vec4Array* coords = new osg::Vec4Array(numParticles);
    for( unsigned int v=0; v<coords->size(); ++v )
        (*coords)[v].set(-1,-1,-1,0);

    ptclGeom->setVertexArray(coords);
    ptclGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));
    ptclGeom->addHandle( "PTCL_BUFFER" );

    // Add particles
    geode->addDrawable( ptclGeom.get() );

    ////////////
    // SPRITE //
    ////////////
    // increase point size within shader
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);

    osg::PointSprite* sprite = new osg::PointSprite();
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);

    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );

    ////////////
    // SHADER //
    ////////////
    // Add program
    osg::Program* program = new osg::Program;
    program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile("osgParticleDemo/shader/PtclSprite.vsh")));
    program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile("osgParticleDemo/shader/PtclSprite.fsh")));
    geode->getOrCreateStateSet()->setAttribute(program);

    // Screen resolution for particle sprite
    osg::Uniform* pixelsize = new osg::Uniform();
    pixelsize->setName( "pixelsize" );
    pixelsize->setType( osg::Uniform::FLOAT_VEC2 );
    pixelsize->set( osg::Vec2(1.0f,50.0f) );
    geode->getOrCreateStateSet()->addUniform( pixelsize );
    geode->setCullingActive( false );

    return geode;
}

int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

    osg::Vec3 bbmin(0,0,0);
    osg::Vec3 bbmax(4,4,4);
    unsigned int numParticles = 64000;

	///////////////
	// RESOURCES //
	///////////////
	// Seeds
	osg::FloatArray* seedValues = new osg::FloatArray();
	for( unsigned int s=0; s<numParticles; ++s )
		seedValues->push_back( float(rand()) / RAND_MAX );

	osgCuda::Buffer* seedBuffer = new osgCuda::Buffer;
	seedBuffer->setElementSize( sizeof(float) );
	seedBuffer->setName( "ptclSeedBuffer" );
	seedBuffer->setDimension(0,numParticles);
	seedBuffer->setArray( seedValues );
	seedBuffer->addHandle( "PTCL_SEEDS" );

	///////////////////
	// SETUP MODULES //
	///////////////////
	// Create module
	PtclDemo::PtclMover* ptclMover = new PtclDemo::PtclMover;
	ptclMover->setName( "ptclMover" );

	PtclDemo::PtclEmitter* ptclEmitter = new PtclDemo::PtclEmitter;
	ptclEmitter->setName( "ptclEmitter" );
	ptclEmitter->setSeedBox( bbmin, bbmax );

	osgCuda::Computation* computation = new osgCuda::Computation;
	computation->setName("computation");
	computation->addModule( *ptclEmitter );    
	computation->addModule( *ptclMover );
	computation->addResource( *seedBuffer );
	// Particles are located in the subgraph of
	// the modules
	computation->addChild( getGeode( numParticles ) );

    /////////////////
    // SETUP SCENE //
    /////////////////
    osg::Group* scene = new osg::Group;
    scene->addChild( computation );
    scene->addChild( getBoundingBox( bbmin, bbmax ) );

    //////////////////
    // SETUP VIEWER //
    //////////////////
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);
    viewer.getCamera()->setComputeNearFarMode( osg::Camera::DO_NOT_COMPUTE_NEAR_FAR );
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );
    viewer.setUpViewInWindow( 50, 50, 640, 480);

	// You must use the single threaded version since osgCompute currently
	// does only support single threaded applications. Please ask in the
	// forum for the multi-threaded version if you need it.
    viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);

    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);

    return viewer.run();
}
