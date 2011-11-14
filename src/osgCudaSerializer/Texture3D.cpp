#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCuda/Texture>
#include "Util.h"

//------------------------------------------------------------------------------
static bool checkIdentifiers( const osgCuda::Texture3D& interopObject )
{
	if( !interopObject.getIdentifiers().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeIdentifiers( osgDB::OutputStream& os, const osgCuda::Texture3D& interopObject )
{	
	const osgCompute::IdentifierSet ids = interopObject.getIdentifiers();
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
static bool readIdentifiers( osgDB::InputStream& is, osgCuda::Texture3D& interopObject )
{

	unsigned int numIds = 0;  
	is >> numIds >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numIds; ++i )
	{
		std::string curId;
        is.readWrappedString( curId );
        curId = osgCuda::trim( curId );

		interopObject.addIdentifier( curId );
	}

	is >> osgDB::END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Texture3D,
						new osgCuda::Texture3D,
						osgCuda::Texture3D,
						"osg::Object osg::StateAttribute osg::Texture osg::Texture3D osgCuda::Texture3D" )
{
	ADD_USER_SERIALIZER( Identifiers );
}
