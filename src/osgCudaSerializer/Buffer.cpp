#include <osg/io_utils>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/ParameterOutput>
#include <osgCuda/Buffer>
#include "Util.h"

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Buffer,
						new osgCuda::Buffer,
						osgCuda::Buffer,
						"osg::Object osgCompute::Resource osgCompute::Memory osgCuda::Buffer" )
{
	ADD_IMAGE_SERIALIZER( Image, osg::Image, NULL );
}

