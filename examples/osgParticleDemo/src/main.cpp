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
#include <osg/AlphaFunc>
#include <osg/PolygonMode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCompute/Program>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>
#include <osgCuda/Computation>
#include <osgCudaStats/Stats>
#include <osgCudaInit/Init>

//////////////////
// COMPUTATIONS //
//////////////////
extern "C" void move( 
                     unsigned int numPtcls, 
                     void* ptcls, 
                     float etime );

class MovePtcls : public osgCompute::Program 
{
public:
    MovePtcls( osg::FrameStamp& fs ) : _fs(&fs) {}

    virtual void launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "MovePtcls");
        }

        float time = (float)_fs->getSimulationTime();
        if( _firstFrame )
        {
            _lastTime = time;
            _firstFrame = false;
        }
        float elapsedtime = static_cast<float>(time - _lastTime);
        _lastTime = time;

        _timer->start();

        move( 
            _ptcls->getNumElements(), 
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ), 
            elapsedtime  );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PARTICLE BUFFER" ) )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    osg::ref_ptr<osgCuda::Timer>        _timer;
    double                              _lastTime;
    bool						        _firstFrame;
    osg::ref_ptr<osg::FrameStamp>       _fs;
    osg::ref_ptr<osgCompute::Memory>    _ptcls;
};


extern "C" void emit(
                     unsigned int numPtcls, 
                     void* ptcls, 
                     void* seeds,  
                     unsigned int seedIdx, 
                     osg::Vec3f bbmin, 
                     osg::Vec3f bbmax );

class EmitPtcls : public osgCompute::Program 
{
public:
    EmitPtcls( osg::Vec3f min, osg::Vec3f max ) : _min(min), _max(max) {}

    virtual void launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_seeds.valid() )
        {
            _seeds = new osgCuda::Buffer;
            _seeds->setElementSize( sizeof(float) );
            _seeds->setName( "Seeds" );
            _seeds->setDimension(0,_ptcls->getNumElements());

            float* seedsData = (float*)_seeds->map(osgCompute::MAP_HOST_TARGET);
            for( unsigned int s=0; s<_ptcls->getNumElements(); ++s )
                seedsData[s] = ( float(rand()) / RAND_MAX );
        }

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "EmitPtcls");
        }

        _timer->start();

        emit(
            _ptcls->getNumElements(),
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ),
            _seeds->map( osgCompute::MAP_DEVICE_SOURCE ),
            (unsigned int)(rand()),
            _min,
            _max  );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PARTICLE BUFFER" ) )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    osg::Vec3f                                        _max;
    osg::Vec3f                                        _min;

    osg::ref_ptr<osgCuda::Timer>                      _timer;
    osg::ref_ptr<osgCompute::Memory>                  _ptcls;
    osg::ref_ptr<osgCompute::Memory>                  _seeds;
};



////////////////////////
// PARTICLE OPERATION //
////////////////////////
class PtclOperation : public osg::Operation
{
public:
    PtclOperation( osg::FrameStamp& fs, osg::ref_ptr<osgCompute::Memory> ptcls, osg::Vec3f bbmin, osg::Vec3f bbmax ) 
    { 
        setKeep( true ); 

        _move = new MovePtcls( fs );
        _move->acceptResource( *ptcls );

        _emit = new EmitPtcls(bbmin,bbmax); 
        _emit->acceptResource( *ptcls );
    }

    virtual void operator() (osg::Object*)
    {
        if( osgCompute::GLMemory::getContext() == NULL || osgCompute::GLMemory::getContext()->getState() == NULL )
            return;

        _emit->launch();
        _move->launch();
    }

public:
    osg::ref_ptr<osgCompute::Program> _emit;
    osg::ref_ptr<osgCompute::Program> _move;
};

//------------------------------------------------------------------------------
osg::ref_ptr<osg::Node> setupScene( osg::ref_ptr<osg::Geometry> geom, osg::Vec3f bbmin, osg::Vec3f bbmax )
{
    osg::ref_ptr<osg::Group> scene = new osg::Group;

    //////////////////////////////
    // CREATE PARTICLE GEOMETRY //
    //////////////////////////////
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable( geom.get() );

    osg::ref_ptr<osg::Program> computation = new osg::Program;

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
    computation->addShader( new osg::Shader(osg::Shader::VERTEX, vtxShader ) );

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

    computation->addShader( new osg::Shader( osg::Shader::FRAGMENT, frgShader ) );
    geode->getOrCreateStateSet()->setAttribute(computation);
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );
    geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "pixelsize", osg::Vec2(1.0f,50.0f) ) ); 
    geode->setCullingActive( false );
    scene->addChild( geode );

    /////////////////////////
    // CREATE BOUNDING BOX //
    /////////////////////////
    osg::Geode* bbox = new osg::Geode;
    bbox->addDrawable(new osg::ShapeDrawable(new osg::Box((bbmin + bbmax) * 0.5f,bbmax.x() - bbmin.x(),bbmax.y() - bbmin.y(),bbmax.z() - bbmin.z()),new osg::TessellationHints()));
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    bbox->getOrCreateStateSet()->setAttribute( new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::LINE));
    scene->addChild( bbox );

    return scene;
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
    viewer.addEventHandler(new osgViewer::HelpHandler);
    // Use 'c'-key to display CUDA stats
    viewer.addEventHandler(new osgCuda::StatsHandler);

    //////////////////
    // SETUP VIEWER //
    //////////////////
    osgCuda::setupOsgCudaAndViewer( viewer );

    /////////////////////
    // PARTICLE BUFFER //
    /////////////////////
    unsigned int numPtcls = 64000;
    osg::ref_ptr<osgCuda::Geometry> geom = new osgCuda::Geometry;
    geom->setName("Particles");
    geom->addIdentifier( "PARTICLE BUFFER" );
    osg::Vec4Array* coords = new osg::Vec4Array(numPtcls);
    for( unsigned int v=0; v<coords->size(); ++v )
        (*coords)[v].set(-1,-1,-1,0);
    geom->setVertexArray(coords);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));

    /////////////////
    // SETUP SCENE //
    /////////////////
    osg::Vec3f bbmin = osg::Vec3f(0,0,0);
    osg::Vec3f bbmax = osg::Vec3f(4,4,4);

    // In this example we use an osg::Operation to update the
    // particle geometry during the Update Traversal.
    viewer.addUpdateOperation( 
        new PtclOperation(*viewer.getFrameStamp(),geom->getMemory(),bbmin,bbmax) );
    viewer.setSceneData( 
        setupScene(geom,bbmin,bbmax) );

    ///////////////
    // RUN SCENE //
    ///////////////
    return viewer.run();
}
