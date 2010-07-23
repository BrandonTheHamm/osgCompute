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
#include <osgCompute/Module>

namespace osgCompute
{   
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	bool Module::existsModule( const std::string& libraryName )
	{
		std::string curLibraryName = osgDB::Registry::instance()->createLibraryNameForNodeKit( libraryName );
		std::string fullPath = osgDB::findLibraryFile( curLibraryName );
		if( fullPath.empty() )
			return false;

		return true;
	}

	//------------------------------------------------------------------------------
	Module* Module::loadModule( const std::string& libraryName )
	{
		std::string curLibraryName = osgDB::Registry::instance()->createLibraryNameForNodeKit( libraryName );
		
		if( osgDB::Registry::instance()->loadLibrary( curLibraryName ) ==  osgDB::Registry::NOT_LOADED )
		{
			osg::notify(osg::WARN)
				<<" Module::loadModule(): cannot find module library "
				<< libraryName <<"."<<std::endl;

			return NULL;
		}

		osg::ref_ptr<osgDB::DynamicLibrary> moduleLibrary = osgDB::Registry::instance()->getLibrary( curLibraryName );
		if( !moduleLibrary.valid() )
		{
			osg::notify(osg::WARN)
				<<" Module::loadModule(): cannot receive module library "
				<< libraryName << " after load." << std::endl;

			return NULL;
		}

		OSGCOMPUTE_CREATE_MODULE_FUNCTION_PTR createModuleFunc = (OSGCOMPUTE_CREATE_MODULE_FUNCTION_PTR) moduleLibrary->getProcAddress( OSGCOMPUTE_CREATE_MODULE_FUNCTION_STR );
		if( createModuleFunc == NULL )
		{
			osg::notify(osg::WARN)
				<<" Module::loadModule(): cannot get pointer to function \""<< OSGCOMPUTE_CREATE_MODULE_FUNCTION_STR<<"\" within module library "
				<< libraryName << "." << std::endl;

			return NULL;
		}

		Module* loadedModule = (*createModuleFunc)();
		if( loadedModule && loadedModule->getLibraryName().empty() )
			loadedModule->setLibraryName( libraryName );

		return loadedModule;
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool Module::init()
    {
        return Resource::init();
    }

    //------------------------------------------------------------------------------
    void Module::setUpdateCallback( ModuleCallback* uc ) 
    { 
        _updateCallback = uc; 
    }

    //------------------------------------------------------------------------------
    ModuleCallback* Module::getUpdateCallback() 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ModuleCallback* Module::getUpdateCallback() const 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Module::setEventCallback( ModuleCallback* ec ) 
    { 
        _eventCallback = ec; 
    }

    //------------------------------------------------------------------------------
    ModuleCallback* Module::getEventCallback() 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ModuleCallback* Module::getEventCallback() const 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Module::acceptResource( Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    bool Module::usesResource( const std::string& handle ) const 
    { 
        return false; 
    }

    //------------------------------------------------------------------------------
    void Module::removeResource( const std::string& handle ) 
    {
    }

    //------------------------------------------------------------------------------
    void Module::removeResource( const Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    Resource* Module::getResource( const std::string& handle ) 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    const Resource* Module::getResource( const std::string& handle ) const 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    void Module::getResources( ResourceList& resourceList, const std::string& handle ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Module::getAllResources( ResourceList& resourceList ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Module::enable() 
    { 
        _enabled = true; 
    }

    //------------------------------------------------------------------------------
    void Module::disable() 
    { 
        _enabled = false; 
    }

    //------------------------------------------------------------------------------
    bool Module::isEnabled() const
    {
        return _enabled;
    }

	//------------------------------------------------------------------------------
	const std::string& Module::getLibraryName() const
	{
		return _libraryName;
	}

	//------------------------------------------------------------------------------
	std::string& Module::getLibraryName()
	{
		return _libraryName;
	}

	//------------------------------------------------------------------------------
	void Module::setLibraryName( const std::string& libraryName )
	{
		_libraryName = libraryName;
	}

    //------------------------------------------------------------------------------
    void Module::clear()
    {
        clearLocal();
        Resource::clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clearLocal()
    {
        _eventCallback = NULL;
        _updateCallback = NULL;
        _enabled = true;
        _clear = true;
    }

}
