#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCuda/Geometry>
#include "Util.h"

//------------------------------------------------------------------------------
static bool checkIdentifiers( const osgCuda::Geometry& geometry )
{
	if( !geometry.getIdentifiers().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeIdentifiers( osgDB::OutputStream& os, const osgCuda::Geometry& geometry )
{	
	const osgCompute::IdentifierSet ids = geometry.getIdentifiers();
	os << (unsigned int)ids.size() << osgDB::BEGIN_BRACKET << std::endl;

	for( osgCompute::IdentifierSetCnstItr idItr = ids.begin();
		idItr != ids.end();
		++idItr )
	{
		os.writeWrappedString( (*idItr) );
		os << std::endl;
	}

	os << osgDB::END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readIdentifiers( osgDB::InputStream& is, osgCuda::Geometry& geometry )
{

	unsigned int numIds = 0;  
	//is >> osgDB::PROPERTY("Identifiers") >> numIds >> osgDB::BEGIN_BRACKET;
	is >> numIds >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numIds; ++i )
	{
		std::string curId;
        is.readWrappedString( curId );
        curId = osgCuda::trim( curId );

		geometry.addIdentifier( curId );
	}

	is >> osgDB::END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Geometry,
						new osgCuda::Geometry,
						osgCuda::Geometry,
						"osg::Object osg::Drawable osg::Geometry osgCuda::Geometry" )
{
	ADD_USER_SERIALIZER( Identifiers );
}


