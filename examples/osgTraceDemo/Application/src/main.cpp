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
#include <osg/BufferObject>
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Program>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>
#include <osgCudaStats/Stats>

//------------------------------------------------------------------------------
osg::ref_ptr<osg::Node> setupScene( unsigned int numParticles, osg::Vec3f bbmin, osg::Vec3f bbmax )
{
    osg::ref_ptr<osg::Group> scene = new osg::Group;

    //////////////
    // GEOMETRY //
    //////////////
    osg::Geode* geode = new osg::Geode;
    osg::ref_ptr<osgCuda::Geometry> ptclGeom = new osgCuda::Geometry;
    // Please note that computations will 
    // use this name in order to identify 
    // it as the particle buffer.
    ptclGeom->setName("Particles");
    ptclGeom->addIdentifier("PTCL_BUFFER");

    osg::Vec4Array* coords = new osg::Vec4Array(numParticles);
    for( unsigned int v=0; v<coords->size(); ++v )
        (*coords)[v].set(-1,-1,-1,1);

    ptclGeom->setVertexArray(coords);
    ptclGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));
    geode->addDrawable( ptclGeom.get() );
    scene->addChild( geode );

    ///////////
    // STATE //
    ///////////
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite(), osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );
    geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "pixelsize", osg::Vec2(1.0f,50.0f) ) );
    geode->setCullingActive( false );

    ////////////
    // SHADER //
    ////////////
    osg::Program* program = new osg::Program;

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

    //////////
    // BBOX //
    //////////
    osg::Geode* bbox = new osg::Geode;
    bbox->addDrawable(new osg::ShapeDrawable(new osg::Box((bbmin + bbmax) * 0.5f,bbmax.x() - bbmin.x(),bbmax.y() - bbmin.y(),bbmax.z() - bbmin.z()),new osg::TessellationHints()));
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    bbox->getOrCreateStateSet()->setAttribute( new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::LINE));
    scene->addChild( bbox );

    return scene;
}

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::Program> setupProgram()
{
    // Execute the program during the update traversal (default) before the subgraph is handled. 
    osgCompute::Program::ComputeOrder order = osgCompute::Program::UPDATE_BEFORECHILDREN;

    osg::ref_ptr<osgCompute::Program> programEmitter = new osgCuda::Program;
    programEmitter->setName( "emit particles program" );
    programEmitter->setComputeOrder( order );
    osgCompute::Computation* ptclEmitter = osgCompute::Computation::loadComputation("osgcuda_ptclemitter");
    if( ptclEmitter )  programEmitter->addComputation( *ptclEmitter );

    osg::ref_ptr<osgCompute::Program> programTracer = new osgCuda::Program;
    programTracer->setName( "trace particles program" );
    programTracer->setComputeOrder( order );
    osgCompute::Computation* ptclTracer = osgCompute::Computation::loadComputation("osgcuda_ptcltracer");
    if( ptclTracer )  programTracer->addComputation( *ptclTracer );
    programTracer->addChild( programEmitter );

    // Write this program to file
    // Please note that you have to copy the generated "tracedemo.osgt" manually to your data-path
    //osgDB::writeNodeFile( *programTracer, "tracedemo.osgt" );

    return programTracer;
}

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::Program> loadProgram()
{
    osg::ref_ptr<osgCompute::Program> program;

    std::string dataFile = osgDB::findDataFile( "osgTraceDemo/scenes/tracedemo.osgt" );
    if( !dataFile.empty() )
        program = dynamic_cast<osgCompute::Program*>( osgDB::readNodeFile( dataFile ) );

    if( !program.valid() ) program = setupProgram();
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
    viewer.getCamera()->setComputeNearFarMode( osg::Camera::DO_NOT_COMPUTE_NEAR_FAR );
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );
    viewer.setUpViewInWindow( 50, 50, 640, 480);
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
    // Please note for loading the computations 
    // "osgcuda_ptclemitter" and "osgcuda_ptcltracer" 
    // all packages must be INSTALLED first. 
    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild( loadProgram() );
    root->addChild( setupScene(64000, osg::Vec3(-1.f,-1.f,-1.f), osg::Vec3(1.f,1.f,1.f)) );
    viewer.setSceneData( root );

    //////////////////////
    // RESOURCE VISITOR //
    //////////////////////
    // The resource-visitor will collect and distribute all resources 
    // of the scene using the names of OpenGL resources as identifiers for modules.
    // In this example "PTCL_BUFFER" is distributed to computations.
    osg::ref_ptr<osgCompute::ResourceVisitor> rv = new osgCompute::ResourceVisitor;
    rv->apply( *root );

    ///////////////
    // RUN SCENE //
    ///////////////
    return viewer.run();
}
