#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCuda/Geometry>

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
static bool readIdentifiers( osgDB::InputStream& is, osgCuda::Geometry& geometry )
{

	unsigned int numIds = 0;  
	//is >> osgDB::PROPERTY("Identifiers") >> numIds >> osgDB::BEGIN_BRACKET;
	is >> numIds >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numIds; ++i )
	{
		std::string curId;
		is.readWrappedString( curId );
		geometry.addIdentifier( curId );
	}

	is >> osgDB::END_BRACKET;
	return true;
}

//------------------------------------------------------------------------------
static bool checkUsage( const osgCuda::Geometry& geometry )
{
	if( geometry.getUsage() != osgCompute::GL_SOURCE_COMPUTE_SOURCE )
		return true;
	else
		return false;
}

//------------------------------------------------------------------------------
static bool writeUsage( osgDB::OutputStream& os, const osgCuda::Geometry& geometry )
{	
	//os << osgDB::PROPERTY("Usage");

	switch( geometry.getUsage() )
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
static bool readUsage( osgDB::InputStream& is, osgCuda::Geometry& geometry )
{
	//is >> osgDB::PROPERTY("Usage");

	std::string interopUsage;
	is.readWrappedString(interopUsage);
	if( interopUsage == "GL_TARGET_COMPUTE_SOURCE" )
		geometry.setUsage( osgCompute::GL_TARGET_COMPUTE_SOURCE );
	else if( interopUsage == "GL_SOURCE_COMPUTE_TARGET" )
		geometry.setUsage( osgCompute::GL_SOURCE_COMPUTE_TARGET );
	else if( interopUsage == "GL_TARGET_COMPUTE_TARGET" )
		geometry.setUsage( osgCompute::GL_TARGET_COMPUTE_TARGET );
	else
		geometry.setUsage( osgCompute::GL_SOURCE_COMPUTE_SOURCE );

	return true;
}

//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Geometry,
						new osgCuda::Geometry,
						osgCuda::Geometry,
						"osg::Object osg::Drawable osg::Geometry osgCuda::Geometry" )
{
	ADD_USER_SERIALIZER( Usage );
	ADD_USER_SERIALIZER( Identifiers );
}


