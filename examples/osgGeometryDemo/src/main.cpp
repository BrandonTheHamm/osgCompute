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
#include <osgCuda/Program>
#include <osgCuda/Geometry>
#include <osgCuda/Memory>
#include <osgCudaStats/Stats>


//------------------------------------------------------------------------------
extern "C" void warp(
          unsigned int numBlocks, 
          unsigned int numThreads, 
          void* vertices,
          unsigned int numVertices,
          void* initPos,
          void* initNormals,
          float simTime );

/**
*/
class Warp : public osgCompute::Computation 
{
public:
    // Use the init function to initialize a module once.
    // You can also setup internal resources here.
    virtual bool init()
    {
        if( !_vertices.valid() )
        {
            osg::notify( osg::WARN )
                << "Warp::init(): buffer is missing."
                << std::endl;

            return false;
        }


        _timer = new osgCuda::Timer;
        _timer->setName( "Warp");
        _timer->init();

        // We need to read the geometry information.
        osgCuda::Geometry* geometry = dynamic_cast<osgCuda::Geometry*>( ((osgCompute::GLMemory*)_vertices.get())->getAdapter() );
        if( !geometry )
            return false;

        // Create the static reference buffers
        osg::ref_ptr<osgCuda::Memory> normals= new osgCuda::Memory;
        normals->setName( "NORMALS" );
        normals->setElementSize( geometry->getNormalArray()->getDataSize() * sizeof(float) );
        normals->setDimension( 0, geometry->getNormalArray()->getNumElements() );
        if( !normals->init() )
        {
            osg::notify( osg::WARN )
                << "Warp::init(): cannot create normal array."
                << std::endl;

            return false;
        }
        _initNormals = normals;

        osg::ref_ptr<osgCuda::Memory> positions = new osgCuda::Memory;
        positions->setName( "POSITIONS" );
        positions->setElementSize( geometry->getVertexArray()->getDataSize() * sizeof(float) );
        positions->setDimension( 0, geometry->getVertexArray()->getNumElements() );
        if( !positions->init() )
        {
            osg::notify( osg::WARN )
                << "Warp::init(): cannot create position array."
                << std::endl;

            return false;
        }
        _initPos = positions;

        // Map the initial buffers to host memory and
        // copy the vertex data.
        memcpy( _initNormals->map(osgCompute::MAP_HOST_TARGET),
            geometry->getNormalArray()->getDataPointer(),
            _initNormals->getByteSize( osgCompute::MAP_HOST ) );

        memcpy( _initPos->map(osgCompute::MAP_HOST_TARGET),
            geometry->getVertexArray()->getDataPointer(),
            _initPos->getByteSize( osgCompute::MAP_HOST ) );

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        _numBlocks = _vertices->getDimension(0) / 128;
        if( _vertices->getDimension(0) % 128 != 0 )
            _numBlocks+=1;

        _numThreads = 128;
        _simulationTime = 0.0f;
        return osgCompute::Computation::init();
    }

    virtual void launch()
    {
        if( isClear() )
            return;

        _timer->start();

        _simulationTime += 0.01f;

        ///////////////////
        // MOVE VERTICES //
        ///////////////////
        warp(_numBlocks,
            _numThreads,
            _vertices->map(),			
            _vertices->getNumElements(),
            _initPos->map(),
            _initNormals->map(),
            _simulationTime );

        // Read out new vertex position on the CPU or change the values with
        // osgCompute::MAP_HOST_TARGET
        //_verts = (float*)_vertices->map( osgCompute::MAP_HOST_SOURCE );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy( "WARP_GEOMETRY" ) )
            _vertices = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    unsigned int                                      _numBlocks;
    unsigned int                                      _numThreads;
    osg::ref_ptr<osgCuda::Timer>                      _timer;
    osg::ref_ptr<osgCompute::Memory>                  _vertices;
    osg::ref_ptr<osgCompute::Memory>				  _initNormals;
    osg::ref_ptr<osgCompute::Memory>				  _initPos;
    float                                             _simulationTime;
};

//------------------------------------------------------------------------------
osg::ref_ptr<osg::Node> setupScene()
{
	osg::ref_ptr<osgCompute::Program> program = new osgCuda::Program;

	//////////////////
	// COW OSG FILE //
	//////////////////
    osg::ref_ptr<osg::Group> cowModel = dynamic_cast<osg::Group*>( osgDB::readNodeFile("cow.osg") );
    if( !cowModel.valid() ) return program;

	///////////////////////////////////
	// TRANFORM TO OSGCUDA::GEOMETRY //
	///////////////////////////////////
	osg::ref_ptr<osg::Geode> cowGeode = dynamic_cast<osg::Geode*>( cowModel->getChild(0) );
	osg::ref_ptr<osg::Geometry> cowGeometry = dynamic_cast<osg::Geometry*>( cowGeode->getDrawable(0) );
	// Configure osgCuda::Geometry
	osgCuda::Geometry* geometry = new osgCuda::Geometry;
	geometry->setName("GEOMETRY");
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

	program->setComputeOrder( osgCompute::Program::UPDATE_BEFORECHILDREN );
	program->addChild( animTransform );

	//////////////////
	// SETUP MODULE //
	//////////////////
    osg::ref_ptr<osgCompute::Computation> warpComputation = new Warp;
    warpComputation->setLibraryName("osgcuda_warp");
	warpComputation->addIdentifier("osgcuda_warp");
	program->addComputation( *warpComputation );
    program->addResource( *geometry->getMemory() );
    return program;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

    //////////////////
    // SETUP VIEWER //
    //////////////////
    osgViewer::Viewer viewer;
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );
    viewer.addEventHandler(new osgViewer::StatsHandler);
    viewer.addEventHandler(new osgCuda::StatsHandler);
    viewer.addEventHandler(new osgViewer::HelpHandler);

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
    return viewer.run();
}
