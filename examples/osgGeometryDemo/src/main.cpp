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
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Computation>
#include <osgCuda/Geometry>
#include "Warp"

// find a geometry object within the subgraph
osg::ref_ptr<osg::Geode> findCowGeode()
{
	osg::ref_ptr<osg::Geode> cowGeode;

	osg::ref_ptr<osg::Node> cow = osgDB::readNodeFile("cow.osg");
	if( !cow.valid() )
		return cowGeode;

	// get hard coded cow geode
	osg::ref_ptr<osg::Group> group = dynamic_cast<osg::Group*>(cow.get());
	if(group)
		cowGeode = dynamic_cast<osg::Geode*>( group->getChild(0) );

	return cowGeode;
}

int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );
	osg::ArgumentParser arguments(&argc,argv);

    //////////////
    // Geometry //
    //////////////
    // Load "cow.osg" file
    osg::ref_ptr<osg::Geode> cowGeode = findCowGeode();
	if( !cowGeode.valid() )
	{
		osg::notify(osg::NOTICE) << "main(): cannot find \"cow.osg\" data." << std::endl;
		return -1;
	}

	osg::ref_ptr<osg::Geometry> cowGeometry = dynamic_cast<osg::Geometry*>( cowGeode->getDrawable(0) );
	if( !cowGeometry.valid() )
	{
		osg::notify(osg::NOTICE) << "main(): cannot find geometry within data \"cow.osg\"." << std::endl;
		return -1;
	}

	osg::ref_ptr<osg::MatrixTransform> animTransform = new osg::MatrixTransform;
	animTransform->addChild( cowGeode );

	// Rotate the model automatically 
	osg::NodeCallback* nc = new osg::AnimationPathCallback(
		animTransform->getBound().center(),osg::Vec3(0.0f,0.0f,1.0f),osg::inDegrees(45.0f));
	animTransform->setUpdateCallback(nc);

    // Configure cuda geometry object
    osgCuda::Geometry* geometry = new osgCuda::Geometry;
    geometry->setVertexArray( cowGeometry->getVertexArray() );
    geometry->addPrimitiveSet( cowGeometry->getPrimitiveSet(0) );
    geometry->setStateSet( cowGeometry->getOrCreateStateSet() );
    geometry->setTexCoordArray( 0, cowGeometry->getTexCoordArray(0) );
    geometry->setNormalArray( cowGeometry->getNormalArray() );
    geometry->setNormalBinding( cowGeometry->getNormalBinding() );
    geometry->addHandle( "WARP_GEOMETRY");
    // Remove original geometry
    bool retval = cowGeode->replaceDrawable( cowGeometry, geometry );

	// Cow geometry can now be deleted
	cowGeometry = NULL;

	//////////////////
	// MODULE SETUP //
	//////////////////
	GeometryDemo::Warp* warpModule = new GeometryDemo::Warp;
	warpModule->setName( "my cow warping" );
	osgCuda::Computation* computation = new osgCuda::Computation;
	computation->addModule( *warpModule );
    computation->addChild( animTransform );
    
    /////////////////
    // SCENE SETUP //
    /////////////////
    osg::Group* scene = new osg::Group;
    scene->addChild( computation );

    //////////////////
    // VIEWER SETUP //
    //////////////////
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
