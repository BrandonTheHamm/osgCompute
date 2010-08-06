#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/WriteFile>
#include <osgCuda/Array>
#include "Util.h"


//------------------------------------------------------------------------------
static bool checkArray( const osgCuda::Array& array )
{
	if( array.getArray() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeArray( osgDB::OutputStream& os, const osgCuda::Array& array )
{	
	os.writeArray( array.getArray() );
	return true;
}

//------------------------------------------------------------------------------
static bool readArray( osgDB::InputStream& is, osgCuda::Array& array )
{
	array.setArray( is.readArray() );
	return true;
}

//------------------------------------------------------------------------------
static bool checkChannelFormatDesc( const osgCuda::Array& array )
{
	return true;
}

//------------------------------------------------------------------------------
static bool writeChannelFormatDesc( osgDB::OutputStream& os, const osgCuda::Array& array )
{	
	const cudaChannelFormatDesc& desc = array.getChannelFormatDesc();
	os << osgDB::BEGIN_BRACKET << std::endl;
    os << desc.x  << desc.y << desc.z << desc.w << std::endl;
	
	switch( desc.f )
	{
	case cudaChannelFormatKindFloat:
		os.writeWrappedString("cudaChannelFormatKindFloat");
        os << std::endl;
		break;
	case cudaChannelFormatKindSigned:
		os.writeWrappedString("cudaChannelFormatKindSigned");
        os << std::endl;
		break;
	case cudaChannelFormatKindUnsigned:
		os.writeWrappedString("cudaChannelFormatKindUnsigned");
        os << std::endl;
		break;
	default:
		os.writeWrappedString("cudaChannelFormatKindNone");
        os << std::endl;
        break;
	}
	os << osgDB::END_BRACKET << std::endl;
	
	return true;
}

//------------------------------------------------------------------------------
static bool readChannelFormatDesc( osgDB::InputStream& is, osgCuda::Array& array )
{
	cudaChannelFormatDesc desc;
	is >> osgDB::BEGIN_BRACKET;

	is >> desc.x;
	is >> desc.y;
	is >> desc.z;
	is >> desc.w;

	std::string channelFormatKind;
	is.readWrappedString(channelFormatKind);
    channelFormatKind = osgCuda::trim( channelFormatKind );

	if( channelFormatKind == "cudaChannelFormatKindFloat" )
		desc.f = cudaChannelFormatKindFloat;
	else if( channelFormatKind == "cudaChannelFormatKindSigned" )
		desc.f = cudaChannelFormatKindSigned;
	else if( channelFormatKind == "cudaChannelFormatKindUnsigned" )
		desc.f = cudaChannelFormatKindUnsigned;
	else
		desc.f = cudaChannelFormatKindNone;
	array.setChannelFormatDesc( desc );

	is >> osgDB::END_BRACKET;
	return true;
}

REGISTER_OBJECT_WRAPPER(osgCuda_Array,
						new osgCuda::Array,
						osgCuda::Array,
						"osg::Object osgCompute::Resource osgCompute::Memory osgCuda::Array" )
{
	ADD_IMAGE_SERIALIZER( Image, osg::Image, NULL );
	ADD_USER_SERIALIZER( Array );
	ADD_USER_SERIALIZER( ChannelFormatDesc );
}
