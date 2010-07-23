#include <osg/io_utils>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/ParameterOutput>
#include <osgCuda/Buffer>

//------------------------------------------------------------------------------
static bool checkArray( const osgCuda::Buffer& buffer )
{
	if( buffer.getImage() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeArray( osgDB::OutputStream& os, const osgCuda::Buffer& buffer )
{	
	os.writeArray( buffer.getArray() );
	return true;
}

//------------------------------------------------------------------------------
static bool readArray( osgDB::InputStream& is, osgCuda::Buffer& buffer )
{
	buffer.setArray( is.readArray() );
	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Buffer,
						new osgCuda::Buffer,
						osgCuda::Buffer,
						"osg::Object osgCompute::Resource osgCompute::Memory osgCuda::Buffer" )
{
	ADD_IMAGE_SERIALIZER( Image, osg::Image, NULL );
	ADD_USER_SERIALIZER( Array );
}

