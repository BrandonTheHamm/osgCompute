/* osgCompute - Copyright (C) 2008-2009 SVT Group
*                                                                     
* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of
* the License, or (at your option) any later version.
*                                                                     
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of 
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesse General Public License for more details.
*
* The full license is in LICENSE file included with this distribution.
*/

#include <osgDB/Registry>
#include <osgDB/FileUtils>
#include <osgCompute/Program>

namespace osgCompute
{   
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	bool Program::existsProgram( const std::string& libraryName )
	{
		std::string curLibraryName = osgDB::Registry::instance()->createLibraryNameForNodeKit( libraryName );
		std::string fullPath = osgDB::findLibraryFile( curLibraryName );
		if( fullPath.empty() )
			return false;

		return true;
	}

	//------------------------------------------------------------------------------
	Program* Program::loadProgram( const std::string& libraryName )
	{
		std::string curLibraryName = osgDB::Registry::instance()->createLibraryNameForNodeKit( libraryName );
		
		if( osgDB::Registry::instance()->loadLibrary( curLibraryName ) ==  osgDB::Registry::NOT_LOADED )
		{
			osg::notify(osg::WARN)
				<<" Program::loadProgram(): cannot find dynamic library "
				<< libraryName <<"."<<std::endl;

			return NULL;
		}

		osg::ref_ptr<osgDB::DynamicLibrary> programLibrary = osgDB::Registry::instance()->getLibrary( curLibraryName );
		if( !programLibrary.valid() )
		{
			osg::notify(osg::WARN)
				<<__FUNCTION__ << ": cannot find dynamic library "
				<< libraryName << " after load." << std::endl;

			return NULL;
		}

		OSGCOMPUTE_CREATE_PROGRAM_FUNCTION_PTR createProgramFunc = 
            (OSGCOMPUTE_CREATE_PROGRAM_FUNCTION_PTR) programLibrary->getProcAddress( OSGCOMPUTE_CREATE_PROGRAM_FUNCTION_STR );
		if( createProgramFunc == NULL )
		{
			osg::notify(osg::WARN)
				<<__FUNCTION__ << ": cannot get pointer to function \""<< OSGCOMPUTE_CREATE_PROGRAM_FUNCTION_STR<<"\" within program library "
				<< libraryName << "." << std::endl;

			return NULL;
		}

		Program* loadedProgram = (*createProgramFunc)();
		if( loadedProgram && loadedProgram->getLibraryName().empty() )
        {
			loadedProgram->setLibraryName( libraryName );
            loadedProgram->addIdentifier( libraryName );
        }

		return loadedProgram;
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Program::Program() : osgCompute::Resource()
    {
        _enabled = true;
    }

    //------------------------------------------------------------------------------
    void Program::launch()
    {
    }

    //------------------------------------------------------------------------------
    void Program::setUpdateCallback( ProgramCallback* uc ) 
    { 
        _updateCallback = uc; 
    }

    //------------------------------------------------------------------------------
    ProgramCallback* Program::getUpdateCallback() 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ProgramCallback* Program::getUpdateCallback() const 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Program::setEventCallback( ProgramCallback* ec ) 
    { 
        _eventCallback = ec; 
    }

    //------------------------------------------------------------------------------
    ProgramCallback* Program::getEventCallback() 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ProgramCallback* Program::getEventCallback() const 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Program::acceptResource( Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    void Program::acceptResource( Resource& resource, const std::string& resourceIdentifier )
    {
    }

    //------------------------------------------------------------------------------
    bool Program::usesResource( const std::string& handle ) const 
    { 
        return false; 
    }

    //------------------------------------------------------------------------------
    void Program::removeResource( const std::string& handle ) 
    {
    }

    //------------------------------------------------------------------------------
    void Program::removeResource( const Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    Resource* Program::getResource( const std::string& handle ) 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    const Resource* Program::getResource( const std::string& handle ) const 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    void Program::getResources( ResourceList& resourceList, const std::string& handle ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Program::getAllResources( ResourceList& resourceList ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Program::enable() 
    { 
        _enabled = true; 
    }

    //------------------------------------------------------------------------------
    void Program::disable() 
    { 
        _enabled = false; 
    }

    //------------------------------------------------------------------------------
    bool Program::isEnabled() const
    {
        return _enabled;
    }

	//------------------------------------------------------------------------------
	const std::string& Program::getLibraryName() const
	{
		return _libraryName;
	}

	//------------------------------------------------------------------------------
	std::string& Program::getLibraryName()
	{
		return _libraryName;
	}

	//------------------------------------------------------------------------------
	void Program::setLibraryName( const std::string& libraryName )
	{
		_libraryName = libraryName;
	}
}
