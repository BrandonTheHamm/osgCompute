#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCompute/Resource>

//------------------------------------------------------------------------------
static bool checkIdentifiers( const osgCompute::Resource& resource )
{
	if( !resource.getIdentifiers().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeIdentifiers( osgDB::OutputStream& os, const osgCompute::Resource& resource )
{	
	const osgCompute::IdentifierSet ids = resource.getIdentifiers();
	//os << osgDB::PROPERTY("Identifiers") << " " << ids.size() << " " << osgDB::BEGIN_BRACKET;
	os << ids.size() << osgDB::BEGIN_BRACKET;
	
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
static bool readIdentifiers( osgDB::InputStream& is, osgCompute::Resource& resource )
{

	unsigned int numIds = 0;  
	//is >> osgDB::PROPERTY("Identifiers") >> numIds >> osgDB::BEGIN_BRACKET;
	is >> numIds >> osgDB::BEGIN_BRACKET;
	
	for( unsigned int i=0; i<numIds; ++i )
	{
		std::string curId;
		is.readWrappedString( curId );
		resource.addIdentifier( curId );
	}

	is >> osgDB::END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCompute_Resource,
						NULL,
						osgCompute::Resource,
						"osg::Object osgCompute::Resource" )
{
	ADD_USER_SERIALIZER( Identifiers );
}