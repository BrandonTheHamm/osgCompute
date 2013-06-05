#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgCompute/Program>
#include <osgCompute/Memory>
#include <osgCuda/Computation>
#include "Util.h"

BEGIN_USER_TABLE( ComputeOrder, osgCuda::Computation );
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
static bool checkComputeOrder( const osgCuda::Computation& computation )
{
    return true;
}

//------------------------------------------------------------------------------
static bool readComputeOrder( osgDB::InputStream& is, osgCuda::Computation& computation )
{
    int order = readOrderValue(is);
    int orderNumber = 0; is >> orderNumber;
    computation.setComputeOrder( static_cast<osgCuda::Computation::ComputeOrder>(order), orderNumber );
    return true;
}

//------------------------------------------------------------------------------
static bool writeComputeOrder( osgDB::OutputStream& os, const osgCuda::Computation& computation )
{
    writeOrderValue( os, (int)computation.getComputeOrder() );
    os << computation.getComputeOrderNum() << std::endl;
    return true;
}


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
		if( resItr->_serialize )
			numRes++;

	// Write attached resources
	os << numRes << os.BEGIN_BRACKET << std::endl;

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

	os << os.END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readResources( osgDB::InputStream& is, osgCuda::Computation& computation )
{
	unsigned int numRes = 0;  
	is >> numRes >> is.BEGIN_BRACKET;

	for( unsigned int i=0; i<numRes; ++i )
	{
        osg::Object* newRes = is.readObject();
        if( newRes != NULL )
        {
            osgCompute::GLMemoryAdapter* ioo = dynamic_cast<osgCompute::GLMemoryAdapter*>( newRes );
            if( ioo != NULL )
            {
                computation.addResource( *ioo->getMemory() );
            }
            else
            {
		        osgCompute::Resource* curRes = dynamic_cast<osgCompute::Resource*>( newRes );
		        if( curRes != NULL ) computation.addResource( *curRes );
            }

        }
	}

	is >> is.END_BRACKET;
	return true;
}


//------------------------------------------------------------------------------
static bool checkPrograms( const osgCuda::Computation& computation )
{
	if( !computation.getPrograms().empty() ) return true;
	else return false;
}

//------------------------------------------------------------------------------
static bool writePrograms( osgDB::OutputStream& os, const osgCuda::Computation& computation )
{	
	const osgCompute::ProgramList modList = computation.getPrograms();


	// Count attached resources
	unsigned int numMods = 0;
	for( osgCompute::ProgramListCnstItr modItr = modList.begin(); modItr != modList.end(); ++modItr )
		if( !(*modItr)->getLibraryName().empty() )
			numMods++;

	// Write attached resources
	os << numMods << os.BEGIN_BRACKET << std::endl;

	for( osgCompute::ProgramListCnstItr modItr = modList.begin(); modItr != modList.end(); ++modItr )
	{
		if( !(*modItr)->getLibraryName().empty() )
		{
			os.writeWrappedString( (*modItr)->getLibraryName() );
			os << std::endl;
		}
	}

	os << os.END_BRACKET << std::endl;
	return true;
}

//------------------------------------------------------------------------------
static bool readPrograms( osgDB::InputStream& is, osgCuda::Computation& computation )
{
	unsigned int numMods = 0;  
	is >> numMods >> is.BEGIN_BRACKET;

	for( unsigned int i=0; i<numMods; ++i )
	{
		std::string moduleLibraryName;
		is.readWrappedString( moduleLibraryName );
        moduleLibraryName = osgCuda::trim( moduleLibraryName );

		if( !osgCompute::Program::existsProgram(moduleLibraryName) )
		{
			osg::notify(osg::WARN) 
				<<" osgCuda_Computation::readPrograms(): cannot find module library "
				<< moduleLibraryName << "." << std::endl;

			continue;
		}

		osgCompute::Program* module = osgCompute::Program::loadProgram( moduleLibraryName );
		if( module != NULL )
		{
			computation.addProgram( *module );
		}
	}

	is >> is.END_BRACKET;
	return true;
}


//------------------------------------------------------------------------------
REGISTER_OBJECT_WRAPPER(osgCuda_Computation,
						new osgCuda::Computation,
						osgCuda::Computation,
						"osg::Object osg::Node osg::Group osgCuda::Computation" )
{
    ADD_USER_SERIALIZER( ComputeOrder );  // _computeOrder & _computeOrderNum
	ADD_USER_SERIALIZER( Programs );
	ADD_USER_SERIALIZER( Resources );
}

