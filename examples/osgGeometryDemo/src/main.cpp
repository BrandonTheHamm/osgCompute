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
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Program>
#include <osgCuda/Geometry>
#include <osgCuda/Buffer>
#include <osgCudaStats/Stats>
#include <osgCudaInit/Init>


/////////////////
// COMPUTATION //
/////////////////
extern "C" void warp(
                     unsigned int numVertices,
                     void* vertices,
                     void* initPos,
                     void* initNormals,
                     float simTime );

class Warp : public osgCompute::Computation 
{
public:
    Warp() : _simulationTime(0) {}

    virtual void launch()
    {
        if( !_vertices.valid() )
            return;

        if( !_initNormals.valid() || !_initPos.valid() )
        {
            // Create the original reference buffers
            osgCuda::Geometry* geometry = dynamic_cast<osgCuda::Geometry*>( ((osgCompute::GLMemory*)_vertices.get())->getAdapter() );
            if( !geometry )
                return;

            _initNormals= new osgCuda::Buffer;
            _initNormals->setName( "NORMALS" );
            _initNormals->setElementSize( geometry->getNormalArray()->getDataSize() * sizeof(float) );
            _initNormals->setDimension( 0, geometry->getNormalArray()->getNumElements() );
            memcpy( _initNormals->map(osgCompute::MAP_HOST_TARGET),
                geometry->getNormalArray()->getDataPointer(),
                _initNormals->getByteSize( osgCompute::MAP_HOST ) );

            _initPos = new osgCuda::Buffer;
            _initPos->setName( "POSITIONS" );
            _initPos->setElementSize( geometry->getVertexArray()->getDataSize() * sizeof(float) );
            _initPos->setDimension( 0, geometry->getVertexArray()->getNumElements() );
            memcpy( _initPos->map(osgCompute::MAP_HOST_TARGET),
                geometry->getVertexArray()->getDataPointer(),
                _initPos->getByteSize( osgCompute::MAP_HOST ) );
        }

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "Warp");
        }

        _timer->start();

        _simulationTime += 0.01f;

        warp(_vertices->getNumElements(),
            _vertices->map(),	
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
        // We expect only to receive one geometry object
        _vertices = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
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
    osg::ref_ptr<osg::Group> cowModel = dynamic_cast<osg::Group*>( osgDB::readNodeFile("osgGeometryDemo/scenes/cudacow.osgt") );
    if( !cowModel.valid() )
    {
        // Build a osgCuda::Geometry
        cowModel = dynamic_cast<osg::Group*>( osgDB::readNodeFile("cow.osg") );
        if( !cowModel.valid() ) return program;

        osg::ref_ptr<osg::Geode> cowGeode = dynamic_cast<osg::Geode*>( cowModel->getChild(0) );
        osg::ref_ptr<osg::Geometry> cowGeometry = dynamic_cast<osg::Geometry*>( cowGeode->getDrawable(0) );
        // Make an identical copy in order to replace osg::Geometry by a osgCuda::Geometry
        osgCuda::Geometry* geometry = new osgCuda::Geometry;
        geometry->setName("GEOMETRY");
        geometry->setVertexArray( cowGeometry->getVertexArray() );
        geometry->addPrimitiveSet( cowGeometry->getPrimitiveSet(0) );
        geometry->setStateSet( cowGeometry->getOrCreateStateSet() );
        geometry->setTexCoordArray( 0, cowGeometry->getTexCoordArray(0) );
        geometry->setNormalArray( cowGeometry->getNormalArray() );
        geometry->setNormalBinding( cowGeometry->getNormalBinding() );
        cowGeode->replaceDrawable( cowGeometry, geometry );
        cowGeometry = NULL;

        // Uncomment the following line to write a osgCuda::Geometry of the cow model!
        // Please note that you have to copy the generated  "cudacow.osgt" manually to your data-path
        // in order to load it afterwards.
        //osgDB::writeNodeFile( *cowModel, "cudacow.osgt" );
    }

    osg::ref_ptr<osg::Geode> geode = dynamic_cast<osg::Geode*>( cowModel->getChild(0) );
    osg::ref_ptr<osgCuda::Geometry> geometry = dynamic_cast<osgCuda::Geometry*>( geode->getDrawable(0) );

    ///////////////////////
    // SETUP COMPUTATION //
    ///////////////////////
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
