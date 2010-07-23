#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCompute/Module>
#include <osgCuda/Computation>

//------------------------------------------------------------------------------
static bool checkResources( const osgCuda::Computation& computation )
{
	if( !computation.getResources().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeResources( osgDB::OutputStream& os, const osgCuda::Computation& computation )
{	
	const osgCompute::ResourceHandleList resList = computation.getResources();

	// Count attached resources
	unsigned int numRes = 0;
	for( osgCompute::ResourceHandleListCnstItr resItr = resList.begin();
		resItr != resList.end();
		++resItr )
		if( resItr->_attached )
			numRes++;

	// Write attached resources
	//os << osgDB::PROPERTY("Resources") << " " << numRes << " " << osgDB::BEGIN_BRACKET;
	os << numRes << osgDB::BEGIN_BRACKET;

	for( osgCompute::ResourceHandleListCnstItr resItr = resList.begin();
		resItr != resList.end();
		++resItr )
		if( resItr->_attached )
			os.writeObject( (*resItr)._resource.get() );

	os << osgDB::END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readResources( osgDB::InputStream& is, osgCuda::Computation& computation )
{
	unsigned int numRes = 0;  
	//is >> osgDB::PROPERTY("Resources") >> numRes >> osgDB::BEGIN_BRACKET;
	is >> numRes >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numRes; ++i )
	{
		osgCompute::Resource* curRes = dynamic_cast<osgCompute::Resource*>( is.readObject() );
		if( curRes != NULL )
			computation.addResource( *curRes );
	}

	is >> osgDB::END_BRACKET;
	return true;
}


//------------------------------------------------------------------------------
static bool checkModules( const osgCuda::Computation& computation )
{
	if( !computation.getModules().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeModules( osgDB::OutputStream& os, const osgCuda::Computation& computation )
{	
	const osgCompute::ModuleList modList = computation.getModules();


	// Count attached resources
	unsigned int numMods = 0;
	for( osgCompute::ModuleListCnstItr modItr = modList.begin(); modItr != modList.end(); ++modItr )
		if( !(*modItr)->getLibraryName().empty() )
			numMods++;

	// Write attached resources
	//os << osgDB::PROPERTY("Modules") << " " << numRes << " " << osgDB::BEGIN_BRACKET;
	os << numMods << osgDB::BEGIN_BRACKET;

	for( osgCompute::ModuleListCnstItr modItr = modList.begin(); modItr != modList.end(); ++modItr )
	{
		if( !(*modItr)->getLibraryName().empty() )
		{
			os.writeWrappedString( (*modItr)->getLibraryName() );
			os << std::endl;
		}
	}

	os << osgDB::END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readModules( osgDB::InputStream& is, osgCuda::Computation& computation )
{
	unsigned int numMods = 0;  
	//is >> osgDB::PROPERTY("Modules") >> numRes >> osgDB::BEGIN_BRACKET;
	is >> numMods >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numMods; ++i )
	{
		std::string moduleLibraryName;
		is.readWrappedString( moduleLibraryName );

		if( !osgCompute::Module::existsModule(moduleLibraryName) )
		{
			osg::notify(osg::WARN) 
				<<" osgCuda_Computation::readModules(): cannot find module library "
				<< moduleLibraryName << "." << std::endl;

			continue;
		}

		osgCompute::Module* module = osgCompute::Module::loadModule( moduleLibraryName );
		if( module != NULL )
		{
			module->addIdentifier( moduleLibraryName );
			computation.addModule( *module );
		}
	}

	is >> osgDB::END_BRACKET;
	return true;
}


//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Computation,
						new osgCuda::Computation,
						osgCuda::Computation,
						"osg::Object osg::Node osg::Group osgCuda::Computation" )
{
	BEGIN_ENUM_SERIALIZER( ComputeOrder, UPDATE_PRE_TRAVERSAL ) ;
		ADD_ENUM_VALUE( UPDATE_POST_TRAVERSAL );
		ADD_ENUM_VALUE( UPDATE_PRE_TRAVERSAL );
		ADD_ENUM_VALUE( RENDER_PRE_RENDER_PRE_TRAVERSAL );
		ADD_ENUM_VALUE( RENDER_PRE_RENDER_POST_TRAVERSAL );
		ADD_ENUM_VALUE( RENDER_POST_RENDER_POST_TRAVERSAL );
		ADD_ENUM_VALUE( RENDER_POST_RENDER_PRE_TRAVERSAL );
	END_ENUM_SERIALIZER();
	ADD_BOOL_SERIALIZER( AutoCheckSubgraph, false );
	ADD_USER_SERIALIZER( Modules );
	ADD_USER_SERIALIZER( Resources );
}
