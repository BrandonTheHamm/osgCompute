#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCuda/Texture>

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
	os << ids.size() << osgDB::BEGIN_BRACKET << std::endl;

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
		interopObject.addIdentifier( curId );
	}

	is >> osgDB::END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
static bool checkUsage( const osgCuda::Texture3D& interopObject )
{
	if( interopObject.getUsage() != osgCompute::GL_SOURCE_COMPUTE_SOURCE )
		return true;
	else
		return false;
}

//------------------------------------------------------------------------------
static bool writeUsage( osgDB::OutputStream& os, const osgCuda::Texture3D& interopObject )
{	
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
static bool readUsage( osgDB::InputStream& is, osgCuda::Texture3D& interopObject )
{
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
REGISTER_OBJECT_WRAPPER(osgCuda_Texture3D,
						new osgCuda::Texture3D,
						osgCuda::Texture3D,
						"osg::Object osg::StateAttribute osg::Texture osg::Texture3D osgCuda::Texture3D" )
{
	ADD_USER_SERIALIZER( Usage );
	ADD_USER_SERIALIZER( Identifiers );
}
