#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCuda/Texture>
#include "Util.h"

//------------------------------------------------------------------------------
static bool checkIdentifiers( const osgCuda::TextureRectangle& interopObject )
{
	if( !interopObject.getIdentifiers().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeIdentifiers( osgDB::OutputStream& os, const osgCuda::TextureRectangle& interopObject )
{	
	const osgCompute::IdentifierSet ids = interopObject.getIdentifiers();
	os << (unsigned int)ids.size() << os.BEGIN_BRACKET << std::endl;

	for( osgCompute::IdentifierSetCnstItr idItr = ids.begin();
		idItr != ids.end();
		++idItr )
	{
		os.writeWrappedString( (*idItr) );
		os << " ";
	}

	os << os.END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readIdentifiers( osgDB::InputStream& is, osgCuda::TextureRectangle& interopObject )
{

	unsigned int numIds = 0;  
	is >> numIds >> is.BEGIN_BRACKET;

	for( unsigned int i=0; i<numIds; ++i )
	{
		std::string curId;
        is.readWrappedString( curId );
        curId = osgCuda::trim( curId );

		interopObject.addIdentifier( curId );
	}

	is >> is.END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_TextureRectangle,
						new osgCuda::TextureRectangle,
						osgCuda::TextureRectangle,
						"osg::Object osg::StateAttribute osg::Texture osg::TextureRectangle osgCuda::TextureRectangle" )
{
	ADD_USER_SERIALIZER( Identifiers );
}
