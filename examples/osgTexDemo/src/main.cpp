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
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgCuda/Computation>
#include <osgCuda/Array>
#include <osgCuda/Texture>

#include "TexStreamer"


static const char* vertShaderSource = 
"#version 130 \n"
//"in mat4 gl_ModelViewProjectionMatrix; \n"
"in vec4 gl_MultiTexCoord0; \n"
"in vec4 gl_Vertex; \n"
"out vec2 texCoord; \n"
"void main()\n"
"{ texCoord = gl_MultiTexCoord0.xy; \n"
"  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \n"
"}\n";

static const char* fragShaderSource = 
"#version 140\n"
"#extension GL_EXT_gpu_shader4 : enable \n"
"uniform usampler2D texImage; \n"
"in vec2 texCoord; \n"
"out vec4 fragColor; \n"
"void main()\n"
"{ vec4 c = texture2D(texImage, texCoord.xy);"
"  fragColor = c; //vec4( texCoord.x, texCoord.y, 0, 1 ); //vec4(1,0,1,1);// uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";

//static const char* fragShaderSource = 
//"#version 140\n"
//"#extension GL_EXT_gpu_shader4 : enable \n"
//"uniform sampler2D texImage; \n"
//"in vec2 texCoord; \n"
//"out vec4 fragColor; \n"
//"void main()\n"
//"{ vec4 c = texture2D(texImage, texCoord.xy);"
//"  fragColor = c; //vec4( texCoord.x, texCoord.y, 0, 1 ); //vec4(1,0,1,1);// uvec4(gl_Color.xyz * 255.0, 255.0);\n"
//"}\n";

osg::Geode* getTexturedQuad( osg::Texture2D& trgTexture )
{
    osg::Geode* geode = new osg::Geode;
    osg::Vec3 llCorner = osg::Vec3(-0.5,0,-0.5);
    osg::Vec3 width = osg::Vec3(1,0,0);
    osg::Vec3 height = osg::Vec3(0,0,1);

    //////////
    // QUAD //
    //////////
    osg::ref_ptr<osg::Geometry> geom = osg::createTexturedQuadGeometry( llCorner, width, height );
    geode->addDrawable( geom );
    geode->getOrCreateStateSet()->setTextureAttributeAndModes( 0, &trgTexture, osg::StateAttribute::ON );
    geode->getOrCreateStateSet()->addUniform( 
        new osg::Uniform( "texImage", 0 ) );

    ////////////
    // SHADER //
    ////////////
    osg::ref_ptr<osg::Program> program = new osg::Program;
    osg::ref_ptr<osg::Shader> vertShader = new osg::Shader;
    vertShader->setShaderSource( vertShaderSource );
    vertShader->setType( osg::Shader::VERTEX );
    osg::ref_ptr<osg::Shader> fragShader = new osg::Shader;
    fragShader->setShaderSource( fragShaderSource );
    fragShader->setType( osg::Shader::FRAGMENT );
    program->addShader( vertShader );
    program->addShader( fragShader );
    geode->getOrCreateStateSet()->setAttribute( program );

    return geode;
}

osg::Group* init()
{
    osg::Group* scene = NULL;

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

    // for arrays you have to provide 
    // a channel desc!!!!
    cudaChannelFormatDesc srcDesc;
    srcDesc.f = cudaChannelFormatKindUnsigned;
    srcDesc.x = 8;
    srcDesc.y = 8;
    srcDesc.z = 8;
    srcDesc.w = 8;

    osg::ref_ptr<osgCuda::Array> srcArray = new osgCuda::Array;
    srcArray->setElementSize( sizeof(osg::Vec4ub) );
    srcArray->setChannelFormatDesc( srcDesc );
    srcArray->setDimension( 0, srcImage->s() );
    srcArray->setDimension( 1, srcImage->t() );
    srcArray->setImage( srcImage );
    // Mark this buffer as a sry array
    srcArray->addIdentifier( "SRC_ARRAY" );

    osg::ref_ptr< osgCuda::Texture2D > trgTexture = new osgCuda::Texture2D;  
    // Note: GL_RGBA8 Bit format is not yet supported by CUDA, use GL_RGBA8UI_EXT instead.
    // GL_RGBA8UI_EXT requires the additional work of scaling the fragment shader
    // output from 0-1 to 0-255. 
    trgTexture->setInternalFormat( GL_RGBA8UI_EXT );
    trgTexture->setSourceFormat( GL_RGBA_INTEGER_EXT );
    trgTexture->setSourceType( GL_UNSIGNED_BYTE );
    // in case you choose a texture size which is not a multiple of the alignment restriction CUDA 
    // will allocate more memory to fulfill the alignemnt requirements. To get the
    // actual pitch (bytes of a single row) call osgCompute::Memory::getPitch()
    trgTexture->setTextureWidth( srcImage->s());
    trgTexture->setTextureHeight( srcImage->t() );
    trgTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    trgTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    // Mark this buffer as the target buffer of the module
    trgTexture->addIdentifier( "TRG_BUFFER" );

    //////////////////
    // MODULE SETUP //
    //////////////////
    osg::ref_ptr<TexDemo::TexStreamer> texStreamer = new TexDemo::TexStreamer;
    texStreamer->setName( "my texture module" );

    osg::ref_ptr<osgCuda::Computation> computation = new osgCuda::Computation;
    // Execute the computation during the rendering, but before
    // the subgraph is rendered. Default is the execution during
    // the update traversal.
    computation->setComputeOrder(  osgCompute::Computation::RENDER_PRE_RENDER_PRE_TRAVERSAL );
    computation->addModule( *texStreamer );
    computation->addResource( *srcArray );
    // the target texture is located in the subgraph of the computation
    computation->addChild( getTexturedQuad( *trgTexture ) );

    /////////////////
    // SCENE SETUP //
    /////////////////
    scene = new osg::Group;
    //scene->addChild( getTexturedQuad(*trgTexture) );
    scene->addChild( computation );
    return scene;
}


class ViewerHandlerTest : public osgGA::GUIEventHandler
{
public:
    ViewerHandlerTest( osgViewer::Viewer* viewer )
        : osgGA::GUIEventHandler(),
        _viewer(viewer) {}

    virtual bool handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa);

protected:
    osgViewer::Viewer* _viewer;
};

//------------------------------------------------------------------------------ 
bool ViewerHandlerTest::handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa)
{
    if( !_viewer )
        return false;

    ////////////////////////
    // RESTART SIMULATION //
    ////////////////////////
    if( ea.getKey() == 'r' &&
        ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN )
    {
        _viewer->setSceneData( NULL );

        osg::ref_ptr<osg::Group> scene = init();
        _viewer->setSceneData( scene );

    }

    return false;
}


int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::WARN );

    osg::Group* scene = init();

    //////////////////
    // VIEWER SETUP //
    //////////////////
    osg::ArgumentParser arguments(&argc, argv);
    osgViewer::Viewer viewer(arguments);
    viewer.setUpViewInWindow( 50, 50, 640, 480);
    viewer.getCamera()->setClearColor( osg::Vec4(0.15, 0.15, 0.15, 1.0) );

    // You must use single threaded version since osgCompute currently
    // does only support single threaded applications. Please ask in the
    // forum for the multi-threaded version if you need it.
    viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);

    viewer.setSceneData( scene );
    viewer.addEventHandler(new osgViewer::StatsHandler);
    viewer.addEventHandler( new ViewerHandlerTest(&viewer) );

    return viewer.run();
}
