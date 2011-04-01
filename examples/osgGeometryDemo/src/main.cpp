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
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Computation>
#include <osgCuda/Geometry>
#include "Warp"

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::Computation> setupComputation()
{
	osg::ref_ptr<osgCompute::Computation> computationNode = new osgCuda::Computation;

	//////////////////
	// COW OSG FILE //
	//////////////////
    osg::ref_ptr<osg::Group> cowModel = dynamic_cast<osg::Group*>( osgDB::readNodeFile("cow.osg") );
    if( !cowModel.valid() ) return computationNode;

	///////////////////////////////////
	// TRANFORM TO OSGCUDA::GEOMETRY //
	///////////////////////////////////
	osg::ref_ptr<osg::Geode> cowGeode = dynamic_cast<osg::Geode*>( cowModel->getChild(0) );
	osg::ref_ptr<osg::Geometry> cowGeometry = dynamic_cast<osg::Geometry*>( cowGeode->getDrawable(0) );
	// Configure osgCuda::Geometry
	osgCuda::Geometry* geometry = new osgCuda::Geometry;
	geometry->setName("dynamic cow geometry");
	geometry->setVertexArray( cowGeometry->getVertexArray() );
	geometry->addPrimitiveSet( cowGeometry->getPrimitiveSet(0) );
	geometry->setStateSet( cowGeometry->getOrCreateStateSet() );
	geometry->setTexCoordArray( 0, cowGeometry->getTexCoordArray(0) );
	geometry->setNormalArray( cowGeometry->getNormalArray() );
	geometry->setNormalBinding( cowGeometry->getNormalBinding() );
	geometry->addIdentifier( "WARP_GEOMETRY");
	// Remove original osg::Geometry
	cowGeode->replaceDrawable( cowGeometry, geometry );
	cowGeometry = NULL;

	///////////////////////
	// SETUP COMPUTATION //
	///////////////////////
	// Rotate the model automatically 
	osg::ref_ptr<osg::MatrixTransform> animTransform = new osg::MatrixTransform;
	osg::NodeCallback* nc = new osg::AnimationPathCallback(
		animTransform->getBound().center(),osg::Vec3(0.0f,0.0f,1.0f),osg::inDegrees(45.0f));
	animTransform->setUpdateCallback(nc);
	animTransform->addChild( cowModel );

	computationNode->setComputeOrder( osgCompute::Computation::UPDATE_BEFORECHILDREN );
	computationNode->addChild( animTransform );

	//////////////////
	// SETUP MODULE //
	//////////////////
    osg::ref_ptr<osgCompute::Module> warpModule = new GeometryDemo::Warp;
    warpModule->setLibraryName("osgcuda_warp");
	warpModule->addIdentifier("osgcuda_warp");
	computationNode->addModule( *warpModule );
    computationNode->addResource( *geometry->getMemory() );

	// Write this computation to file
	//osgDB::writeNodeFile( *computationNode, "geomdemo.osgt" );

    return computationNode;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );
	osg::ArgumentParser arguments(&argc,argv);
	osgViewer::Viewer viewer(arguments);

	/////////////////
	// COMPUTATION //
	/////////////////
	osg::ref_ptr<osgCompute::Computation> computation = setupComputation();

	osg::ref_ptr<osgCompute::Module> module = computation->getModule( "osgcuda_warp" );
	if( !module )
	{
		osg::notify(osg::FATAL) << "Cannot find module identified by osgcuda_warp." << std::endl;
		return -1;
	}
	module->setUserData( viewer.getFrameStamp() );

    /////////////////
    // SCENE SETUP //
    /////////////////
    osg::Group* scene = new osg::Group;
    scene->addChild( computation );

    //////////////////
    // VIEWER SETUP //
    //////////////////
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );

    // You must use single threaded version since osgCompute currently
    // does only support single threaded applications. Please ask in the
    // forum for the multi-threaded version if you need it.

    viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);
    viewer.setReleaseContextAtEndOfFrameHint(false);
    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);

    return viewer.run();
}
