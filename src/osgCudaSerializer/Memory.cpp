#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCompute/Memory>

//------------------------------------------------------------------------------
static bool checkDimensions( const osgCompute::Memory& memory )
{
	return true;
}

//------------------------------------------------------------------------------
static bool writeDimensions( osgDB::OutputStream& os, const osgCompute::Memory& memory )
{	
	os << memory.getNumDimensions() << osgDB::BEGIN_BRACKET << std::endl;

	for( unsigned int d=0; d<memory.getNumDimensions(); ++d )
	{
		os << memory.getDimension(d) << std::endl;
	}

	os << osgDB::END_BRACKET << std::endl;

	return true;
}

//------------------------------------------------------------------------------
static bool readDimensions( osgDB::InputStream& is, osgCompute::Memory& memory )
{
	unsigned int numDim = 0;  
	is >> numDim >> osgDB::BEGIN_BRACKET;

	for( unsigned int d=0; d<numDim; ++d )
	{
		unsigned int curDimSize;
		is >> curDimSize;
		memory.setDimension(d,curDimSize);
	}

	is >> osgDB::END_BRACKET;

	return true;
}


//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCompute_Memory,
						NULL,
						osgCompute::Memory,
						"osg::Object osgCompute::Resource osgCompute::Memory" )
{
	ADD_UINT_SERIALIZER( ElementSize, 0 );
	ADD_USER_SERIALIZER( Dimensions );
	ADD_UINT_SERIALIZER( AllocHint, 0 );
}

