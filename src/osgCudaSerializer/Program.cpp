#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCompute/Computation>
#include <osgCompute/Memory>
#include <osgCuda/Program>
#include "Util.h"

BEGIN_USER_TABLE( ComputeOrder, osgCuda::Program );
ADD_USER_VALUE( UPDATE_AFTERCHILDREN );
ADD_USER_VALUE( UPDATE_BEFORECHILDREN );
ADD_USER_VALUE( UPDATE_AFTERCHILDREN_NORENDER );
ADD_USER_VALUE( UPDATE_BEFORECHILDREN_NORENDER );
ADD_USER_VALUE( PRE_RENDER );
ADD_USER_VALUE( POST_RENDER );
END_USER_TABLE()

USER_READ_FUNC( ComputeOrder, readOrderValue )
USER_WRITE_FUNC( ComputeOrder, writeOrderValue )

//------------------------------------------------------------------------------
static bool checkComputeOrder( const osgCuda::Program& program )
{
    return true;
}

//------------------------------------------------------------------------------
static bool readComputeOrder( osgDB::InputStream& is, osgCuda::Program& program )
{
    int order = readOrderValue(is);
    int orderNumber = 0; is >> orderNumber;
    program.setComputeOrder( static_cast<osgCuda::Program::ComputeOrder>(order), orderNumber );
    return true;
}

//------------------------------------------------------------------------------
static bool writeComputeOrder( osgDB::OutputStream& os, const osgCuda::Program& program )
{
    writeOrderValue( os, (int)program.getComputeOrder() );
    os << program.getComputeOrderNum() << std::endl;
    return true;
}


//------------------------------------------------------------------------------
static bool checkResources( const osgCuda::Program& program )
{
    if( !program.getResources().empty() ) return true;
    else return false;
}

//------------------------------------------------------------------------------
static bool writeResources( osgDB::OutputStream& os, const osgCuda::Program& program )
{	
	const osgCompute::ResourceHandleList resList = program.getResources();

	// Count attached resources
	unsigned int numRes = 0;
	for( osgCompute::ResourceHandleListCnstItr resItr = resList.begin();
		resItr != resList.end();
		++resItr )
		if( resItr->_serialize )
			numRes++;

	// Write attached resources
	os << numRes << osgDB::BEGIN_BRACKET << std::endl;

	for( osgCompute::ResourceHandleListCnstItr resItr = resList.begin();
		resItr != resList.end();
		++resItr )
    {
		if( resItr->_serialize )
        {
            osgCompute::GLMemory* iom = dynamic_cast<osgCompute::GLMemory*>( (*resItr)._resource.get() );
            if( iom != NULL )
            { // if layered interoperability object then store the interoperability object
                osg::Object* ioo = dynamic_cast<osg::Object*>( iom->getAdapter() );
                if( ioo ) os.writeObject( ioo );
            }
            else
            {
			    os.writeObject( (*resItr)._resource.get() );
            }
        }
    }

	os << osgDB::END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readResources( osgDB::InputStream& is, osgCuda::Program& program )
{
	unsigned int numRes = 0;  
	is >> numRes >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numRes; ++i )
	{
        osg::Object* newRes = is.readObject();
        if( newRes != NULL )
        {
            osgCompute::GLMemoryAdapter* ioo = dynamic_cast<osgCompute::GLMemoryAdapter*>( newRes );
            if( ioo != NULL )
            {
                program.addResource( *ioo->getMemory() );
            }
            else
            {
		        osgCompute::Resource* curRes = dynamic_cast<osgCompute::Resource*>( newRes );
		        if( curRes != NULL ) program.addResource( *curRes );
            }

        }
	}

	is >> osgDB::END_BRACKET;
	return true;
}


//------------------------------------------------------------------------------
static bool checkComputations( const osgCuda::Program& program )
{
	if( !program.getComputations().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writeComputations( osgDB::OutputStream& os, const osgCuda::Program& program )
{	
	const osgCompute::ComputationList modList = program.getComputations();


	// Count attached resources
	unsigned int numMods = 0;
	for( osgCompute::ComputationListCnstItr modItr = modList.begin(); modItr != modList.end(); ++modItr )
		if( !(*modItr)->getLibraryName().empty() )
			numMods++;

	// Write attached resources
	os << numMods << osgDB::BEGIN_BRACKET << std::endl;

	for( osgCompute::ComputationListCnstItr modItr = modList.begin(); modItr != modList.end(); ++modItr )
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
static bool readComputations( osgDB::InputStream& is, osgCuda::Program& program )
{
	unsigned int numMods = 0;  
	is >> numMods >> osgDB::BEGIN_BRACKET;

	for( unsigned int i=0; i<numMods; ++i )
	{
		std::string moduleLibraryName;
		is.readWrappedString( moduleLibraryName );
        moduleLibraryName = osgCuda::trim( moduleLibraryName );

		if( !osgCompute::Computation::existsComputation(moduleLibraryName) )
		{
			osg::notify(osg::WARN) 
				<<" osgCuda_Program::readComputations(): cannot find module library "
				<< moduleLibraryName << "." << std::endl;

			continue;
		}

		osgCompute::Computation* module = osgCompute::Computation::loadComputation( moduleLibraryName );
		if( module != NULL )
		{
			program.addComputation( *module );
		}
	}

	is >> osgDB::END_BRACKET;
	return true;
}


//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Program,
						new osgCuda::Program,
						osgCuda::Program,
						"osg::Object osg::Node osg::Group osgCuda::Program" )
{
    ADD_USER_SERIALIZER( ComputeOrder );  // _computeOrder & _computeOrderNum
	ADD_USER_SERIALIZER( Computations );
	ADD_USER_SERIALIZER( Resources );
}

