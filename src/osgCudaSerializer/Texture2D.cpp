#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCuda/Texture>

//------------------------------------------------------------------------------
static bool checkIdentifiers( const osgCuda::Texture2D& interopObject )
{
	if( !interopObject.getIdentifiers().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeIdentifiers( osgDB::OutputStream& os, const osgCuda::Texture2D& interopObject )
{	
	const osgCompute::IdentifierSet ids = interopObject.getIdentifiers();
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
static bool readIdentifiers( osgDB::InputStream& is, osgCuda::Texture2D& interopObject )
{

	unsigned int numIds = 0;  
	//is >> osgDB::PROPERTY("Identifiers") >> numIds >> osgDB::BEGIN_BRACKET;
	is >> numIds >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numIds; ++i )
	{
		std::string curId;
		is.readWrappedString( curId );
		interopObject.addIdentifier( curId );
	}

	is >> osgDB::END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
static bool checkUsage( const osgCuda::Texture2D& interopObject )
{
	if( interopObject.getUsage() != osgCompute::GL_SOURCE_COMPUTE_SOURCE )
		return true;
	else
		return false;
}

//------------------------------------------------------------------------------
static bool writeUsage( osgDB::OutputStream& os, const osgCuda::Texture2D& interopObject )
{	
	//os << osgDB::PROPERTY("Usage");

	switch( interopObject.getUsage() )
	{
	case osgCompute::GL_TARGET_COMPUTE_SOURCE:
		os.writeWrappedString(" GL_TARGET_COMPUTE_SOURCE ");
		break;
	case osgCompute::GL_SOURCE_COMPUTE_TARGET:
		os.writeWrappedString(" GL_SOURCE_COMPUTE_TARGET ");
		break;
	case osgCompute::GL_TARGET_COMPUTE_TARGET:
		os.writeWrappedString(" GL_TARGET_COMPUTE_TARGET ");
		break;
	default:
		os.writeWrappedString(" GL_SOURCE_COMPUTE_SOURCE ");
	}

	os << std::endl;

	return true;
}

//------------------------------------------------------------------------------
static bool readUsage( osgDB::InputStream& is, osgCuda::Texture2D& interopObject )
{
	//is >> osgDB::PROPERTY("Usage");

	std::string interopUsage;
	is.readWrappedString(interopUsage);
	if( interopUsage == "GL_TARGET_COMPUTE_SOURCE" )
		interopObject.setUsage( osgCompute::GL_TARGET_COMPUTE_SOURCE );
	else if( interopUsage == "GL_SOURCE_COMPUTE_TARGET" )
		interopObject.setUsage( osgCompute::GL_SOURCE_COMPUTE_TARGET );
	else if( interopUsage == "GL_TARGET_COMPUTE_TARGET" )
		interopObject.setUsage( osgCompute::GL_TARGET_COMPUTE_TARGET );
	else
		interopObject.setUsage( osgCompute::GL_SOURCE_COMPUTE_SOURCE );

	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Texture2D,
						new osgCuda::Texture2D,
						osgCuda::Texture2D,
						"osg::Object osg::StateAttribute osg::Texture osg::Texture2D osgCuda::Texture2D" )
{
	ADD_USER_SERIALIZER( Usage );
	ADD_USER_SERIALIZER( Identifiers );
}
