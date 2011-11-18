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
#include <cuda_runtime.h>
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
#include <osgCuda/Program>
#include <osgCuda/Memory>
#include <osgCuda/Geometry>
#include <osgCudaStats/Stats>

//////////////////////
// GLOBAL VARIABLES //
//////////////////////
const osg::Vec3f   bbmin = osg::Vec3f(0,0,0);
const osg::Vec3f   bbmax = osg::Vec3f(4,4,4);
const unsigned int numParticles = 64000;

//////////////////////////
// ADDITIONAL RESOURCES //
//////////////////////////
/**
*/
class AdvanceTime : public osgCompute::Resource
{
public: 
    osg::ref_ptr<osg::FrameStamp> _fs;
};

/**
*/
class EmitterBox : public osgCompute::Resource
{
public:
    osg::Vec3f _min;
    osg::Vec3f _max;
};

//////////////////////
// EXTERN FUNCTIONS //
//////////////////////
//------------------------------------------------------------------------------
extern "C" void reseed(  
              unsigned int numBlocks,
              unsigned int numThreads,
              void* ptcls,
              void* seeds,
              unsigned int seedCount,
              unsigned int seedIdx,
              float3 bbmin,
              float3 bbmax,
              unsigned int numPtcls );

//------------------------------------------------------------------------------
extern "C" void move( 
              unsigned int numBlocks, 
              unsigned int numThreads, 
              void* ptcls, 
              float etime,
              unsigned int numPtcls );

//////////////////
// COMPUTATIONS //
//////////////////
/**
*/
class PtclMover : public osgCompute::Computation 
{
public:
    virtual void launch()
    {
        if( !_ptcls.valid() || !_advanceTime.valid() )
        {
            osg::notify( osg::WARN )
                << "PtclMover::launch(): resources are missing."
                << std::endl;

            return;
        }

        void* ptclPos = _ptcls->map( osgCompute::MAP_DEVICE_TARGET );
        if( ptclPos == NULL  )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "PtclMover");
        }

        _timer->start();

        /////////////////////
        // ADVANCE IN TIME //
        /////////////////////
        float time = (float)(_advanceTime->_fs)->getSimulationTime();
        if( _firstFrame )
        {
            _lastTime = time;
            _firstFrame = false;
        }

        float elapsedtime = static_cast<float>(time - _lastTime);
        _lastTime = time;

        ////////////////////
        // MOVE PARTICLES //
        ////////////////////
        unsigned int numBlocks = (_ptcls->getDimension(0) / 128)+1;
        unsigned int numThreads = 128;

        move( numBlocks, 
              numThreads, 
              ptclPos, 
              elapsedtime,
              _ptcls->getDimension(0) );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );

        if( resource.isIdentifiedBy("PTCL_ADVANCETIME") )
            _advanceTime = dynamic_cast<AdvanceTime*>( &resource );
    }

private:
    osg::ref_ptr<osgCuda::Timer>        _timer;
    double                              _lastTime;
    bool						        _firstFrame;
    osg::ref_ptr<AdvanceTime>           _advanceTime;
    osg::ref_ptr<osgCompute::Memory>    _ptcls;
};

/**
*/
class PtclEmitter : public osgCompute::Computation 
{
public:
    virtual void launch()
    {
        if( !_ptcls.valid() || !_box.valid() )
        {
            osg::notify( osg::WARN )
                << "ParticleEmitter::launch() resources are missing."
                << std::endl;

            return;
        }        
        

        void* ptclPos = _ptcls->map( osgCompute::MAP_DEVICE_TARGET );
        void* seedPos = _seeds->map( osgCompute::MAP_DEVICE_SOURCE );
        if( ptclPos == NULL || seedPos == NULL )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "PtclEmitter");
        }

        _timer->start();

        //////////////////////
        // RESEED PARTICLES //
        //////////////////////
        unsigned int numBlocks = (_ptcls->getDimension(0) / 128) + 1;
        unsigned int numThreads = 128;

        float3 bbmin = { _box->_min.x(), _box->_min.y(), _box->_min.z() };
        float3 bbmax = { _box->_max.x(), _box->_max.y(), _box->_max.z() };

        reseed(
            numBlocks,
            numThreads,
            ptclPos,
            seedPos,
            _seeds->getDimension(0),
            static_cast<unsigned int>(rand()),
            bbmin,
            bbmax,
            _ptcls->getDimension(0) );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy("PTCL_SEEDS") )
            _seeds = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy("EMITTER_BOX") )
            _box = dynamic_cast<EmitterBox*>( &resource );
    }

private:
    osg::ref_ptr<osgCuda::Timer>                      _timer;
    osg::ref_ptr<EmitterBox>                          _box;
    osg::ref_ptr<osgCompute::Memory>                  _ptcls;
    osg::ref_ptr<osgCompute::Memory>                  _seeds;
};

//////////////////////
// UPDATE OPERATION //
//////////////////////
// You do not need to use programs in order to launch modules. In this example
// we use an updaet operation to update particle geometry with CUDA
class PtclOperation : public osg::Operation
{
public:
    virtual void init( osgCompute::ResourceVisitor& rv )
    {
        setKeep( true );

        _emitter = new PtclEmitter; 
        _mover = new PtclMover;

        osgCompute::ResourceSet resources = rv.getResources();
        for( osgCompute::ResourceSetItr itr = resources.begin(); itr != resources.end(); ++itr )
        {
            _emitter->acceptResource( *(*itr) );
            _mover->acceptResource( *(*itr) );
        }
    }

    virtual void operator() (osg::Object*)
    {
        _emitter->launch();
        _mover->launch();
    }

private:
    osg::ref_ptr<osgCompute::Computation> _emitter;
    osg::ref_ptr<osgCompute::Computation> _mover;
};


//////////////////////
// GLOBAL FUNCTIONS //
//////////////////////
//------------------------------------------------------------------------------
osg::ref_ptr<osg::Node> getScene()
{
    osg::ref_ptr<osg::Group> scene = new osg::Group;

    //////////////////////////////
    // CREATE PARTICLE GEOMETRY //
    //////////////////////////////
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    osg::ref_ptr<osgCuda::Geometry> ptclGeom = new osgCuda::Geometry;
    ptclGeom->setName("PARTICLE BUFFER");

    // Initialize the Particles
    osg::Vec4Array* coords = new osg::Vec4Array(numParticles);
    for( unsigned int v=0; v<coords->size(); ++v )
        (*coords)[v].set(-1,-1,-1,0);

    ptclGeom->setVertexArray(coords);
    ptclGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));
    ptclGeom->addIdentifier( "PTCL_BUFFER" );
    geode->addDrawable( ptclGeom.get() );

    // Sprite 
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );

    // Shader 
    osg::ref_ptr<osg::Program> program = new osg::Program;

    const std::string vtxShader=
    "uniform vec2 pixelsize;                                                                \n"
    "                                                                                       \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   vec4 worldPos = vec4(gl_Vertex.x,gl_Vertex.y,gl_Vertex.z,1.0);                      \n"
    "   vec4 projPos = gl_ModelViewProjectionMatrix * worldPos;                             \n"
    "                                                                                       \n"
    "   float dist = projPos.z / projPos.w;                                                 \n"
    "   float distAlpha = (dist+1.0)/2.0;                                                   \n"
    "   gl_PointSize = pixelsize.y - distAlpha * (pixelsize.y - pixelsize.x);               \n"
    "                                                                                       \n"
    "   gl_Position = projPos;                                                              \n"
    "}                                                                                      \n";
    program->addShader( new osg::Shader(osg::Shader::VERTEX, vtxShader ) );

    const std::string frgShader=
    "void main (void)                                                                       \n"
    "{                                                                                      \n"
    "   vec4 result;                                                                        \n"
    "                                                                                       \n"
    "   vec2 tex_coord = gl_TexCoord[0].xy;                                                 \n"
    "   tex_coord.y = 1.0-tex_coord.y;                                                      \n"
    "   float d = 2.0*distance(tex_coord.xy, vec2(0.5, 0.5));                               \n"
    "   result.a = step(d, 1.0);                                                            \n"
    "                                                                                       \n"
    "   vec3 eye_vector = normalize(vec3(0.0, 0.0, 1.0));                                   \n"
    "   vec3 light_vector = normalize(vec3(2.0, 2.0, 1.0));                                 \n"
    "   vec3 surface_normal = normalize(vec3(2.0*                                           \n"
    "           (tex_coord.xy-vec2(0.5, 0.5)), sqrt(1.0-d)));                               \n"
    "   vec3 half_vector = normalize(eye_vector+light_vector);                              \n"
    "                                                                                       \n"
    "   float specular = dot(surface_normal, half_vector);                                  \n"
    "   float diffuse  = dot(surface_normal, light_vector);                                 \n"
    "                                                                                       \n"
    "   vec4 lighting = vec4(0.75, max(diffuse, 0.0), pow(max(specular, 0.0), 40.0), 0.0);  \n"
    "                                                                                       \n"
    "   result.rgb = lighting.x*vec3(0.2, 0.8, 0.2)+lighting.y*vec3(0.6, 0.6, 0.6)+         \n"
    "   lighting.z*vec3(0.25, 0.25, 0.25);                                                  \n"
    "                                                                                       \n"
    "   gl_FragColor = result;                                                              \n"
    "}                                                                                      \n";

    program->addShader( new osg::Shader( osg::Shader::FRAGMENT, frgShader ) );
    geode->getOrCreateStateSet()->setAttribute(program);

    // Screen resolution for particle sprite
    osg::Uniform* pixelsize = new osg::Uniform();
    pixelsize->setName( "pixelsize" );
    pixelsize->setType( osg::Uniform::FLOAT_VEC2 );
    pixelsize->set( osg::Vec2(1.0f,50.0f) );
    geode->getOrCreateStateSet()->addUniform( pixelsize );
    geode->setCullingActive( false );
    scene->addChild( geode );

    /////////////////////////
    // CREATE BOUNDING BOX //
    /////////////////////////
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

    osg::Geode* bbox = new osg::Geode;
    bbox->addDrawable( bbgeom );
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    scene->addChild( bbox );

    return scene;
}

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::ResourceVisitor> getVisitor( osg::FrameStamp* fs )
{
    osg::ref_ptr<osgCompute::ResourceVisitor> rv = new osgCompute::ResourceVisitor;
    
    //////////////////////
    // GLOBAL RESOURCES //
    //////////////////////
    // You can add resources directly to resource visitor.
    // Each resource will be distributed to all programs
    // located in the graph.

    // EMITTER BOX
    osg::ref_ptr<EmitterBox> emitterBox = new EmitterBox;
    emitterBox->addIdentifier( "EMITTER_BOX" );
    emitterBox->_min = bbmin;
    emitterBox->_max = bbmax;
    rv->addResource( *emitterBox );

    // FRAME STAMP
    osg::ref_ptr<AdvanceTime> advanceTime = new AdvanceTime;
    advanceTime->addIdentifier( "PTCL_ADVANCETIME" );
    advanceTime->_fs = fs;
    rv->addResource( *advanceTime );

    // SEED POSITIONS
    osg::Image* seedValues = new osg::Image();
	seedValues->allocateImage(numParticles,1,1,GL_LUMINANCE,GL_FLOAT);
    
	float* seeds = (float*)seedValues->data();
	for( unsigned int s=0; s<numParticles; ++s )
        seeds[s] = ( float(rand()) / RAND_MAX );

    osg::ref_ptr<osgCuda::Memory> seedBuffer = new osgCuda::Memory;
    seedBuffer->setElementSize( sizeof(float) );
    seedBuffer->setName( "PARTICLE SEEDS" );
    seedBuffer->setDimension(0,numParticles);
    seedBuffer->setImage( seedValues );
    seedBuffer->addIdentifier( "PTCL_SEEDS" );
    rv->addResource( *seedBuffer );

    return rv;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);
    viewer.getCamera()->setComputeNearFarMode( osg::Camera::DO_NOT_COMPUTE_NEAR_FAR );
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.addEventHandler(new osgViewer::StatsHandler);
    viewer.addEventHandler(new osgCuda::StatsHandler);
    viewer.addEventHandler(new osgViewer::HelpHandler);

    //////////////////
    // SETUP VIEWER //
    //////////////////
    osgCuda::setupOsgCudaAndViewer( viewer );

    /////////////////
    // SETUP SCENE //
    /////////////////
    viewer.setSceneData( getScene() );

    //////////////////////
    // RESOURCE VISITOR //
    //////////////////////
    // Use a resource visitor to collect and distribute
    // resources among a sub-graph. Here it is applied 
    // to the scene root. In the first pass it collects all resources
    // and in a second traversal it distributes them among the
    // programs in a graph.
    osg::ref_ptr<osgCompute::ResourceVisitor> visitor = getVisitor( viewer.getFrameStamp() );
    visitor->setMode( osgCompute::ResourceVisitor::COLLECT );
    visitor->apply( *viewer.getSceneData() );

    ////////////////////////////
    // SETUP UPDATE OPERATION //
    ////////////////////////////
    // In this example we use an update operation to update the
    // particle geometry over time. Usually if you update
    // sceen graph structures you would like to prefer an osgCompute::Program
    // in order to launch modules.
    osg::ref_ptr<PtclOperation> ptclOperation = new PtclOperation;
    ptclOperation->init( *visitor );
    viewer.addUpdateOperation( ptclOperation );

    ///////////////
    // RUN SCENE //
    ///////////////
    return viewer.run();
}
