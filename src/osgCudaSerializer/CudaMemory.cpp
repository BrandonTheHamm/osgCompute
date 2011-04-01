#include <osg/io_utils>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/ParameterOutput>
#include <osgCuda/Memory>
#include "Util.h"

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Memory,
						new osgCuda::Memory,
						osgCuda::Memory,
						"osg::Object osgCompute::Resource osgCompute::Memory osgCuda::Memory" )
{
	ADD_IMAGE_SERIALIZER( Image, osg::Image, NULL );
}

