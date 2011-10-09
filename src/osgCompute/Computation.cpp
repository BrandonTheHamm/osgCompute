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
#include <osgCompute/Computation>

namespace osgCompute
{   
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	bool Computation::existsComputation( const std::string& libraryName )
	{
		std::string curLibraryName = osgDB::Registry::instance()->createLibraryNameForNodeKit( libraryName );
		std::string fullPath = osgDB::findLibraryFile( curLibraryName );
		if( fullPath.empty() )
			return false;

		return true;
	}

	//------------------------------------------------------------------------------
	Computation* Computation::loadComputation( const std::string& libraryName )
	{
		std::string curLibraryName = osgDB::Registry::instance()->createLibraryNameForNodeKit( libraryName );
		
		if( osgDB::Registry::instance()->loadLibrary( curLibraryName ) ==  osgDB::Registry::NOT_LOADED )
		{
			osg::notify(osg::WARN)
				<<" Computation::loadComputation(): cannot find dynamic library "
				<< libraryName <<"."<<std::endl;

			return NULL;
		}

		osg::ref_ptr<osgDB::DynamicLibrary> computationLibrary = osgDB::Registry::instance()->getLibrary( curLibraryName );
		if( !computationLibrary.valid() )
		{
			osg::notify(osg::WARN)
				<<" Computation::loadComputation(): cannot find dynamic library "
				<< libraryName << " after load." << std::endl;

			return NULL;
		}

		OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION_PTR createComputationFunc = 
            (OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION_PTR) computationLibrary->getProcAddress( OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION_STR );
		if( createComputationFunc == NULL )
		{
			osg::notify(osg::WARN)
				<<" Computation::loadComputation(): cannot get pointer to function \""<< OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION_STR<<"\" within computation library "
				<< libraryName << "." << std::endl;

			return NULL;
		}

		Computation* loadedComputation = (*createComputationFunc)();
		if( loadedComputation && loadedComputation->getLibraryName().empty() )
        {
			loadedComputation->setLibraryName( libraryName );
            loadedComputation->addIdentifier( libraryName );
        }

		return loadedComputation;
	}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool Computation::init()
    {
        return Resource::init();
    }

    //------------------------------------------------------------------------------
    void Computation::launch()
    {
    }

    //------------------------------------------------------------------------------
    void Computation::setLaunchHint( unsigned int state )
    {
    }

    //------------------------------------------------------------------------------
    unsigned int Computation::getLaunchHint() const
    {
        return 0;
    }

    //------------------------------------------------------------------------------
    void Computation::setUpdateCallback( ComputationCallback* uc ) 
    { 
        _updateCallback = uc; 
    }

    //------------------------------------------------------------------------------
    ComputationCallback* Computation::getUpdateCallback() 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ComputationCallback* Computation::getUpdateCallback() const 
    { 
        return _updateCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Computation::setEventCallback( ComputationCallback* ec ) 
    { 
        _eventCallback = ec; 
    }

    //------------------------------------------------------------------------------
    ComputationCallback* Computation::getEventCallback() 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    const ComputationCallback* Computation::getEventCallback() const 
    { 
        return _eventCallback.get(); 
    }

    //------------------------------------------------------------------------------
    void Computation::acceptResource( Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    bool Computation::usesResource( const std::string& handle ) const 
    { 
        return false; 
    }

    //------------------------------------------------------------------------------
    void Computation::removeResource( const std::string& handle ) 
    {
    }

    //------------------------------------------------------------------------------
    void Computation::removeResource( const Resource& resource ) 
    {
    }

    //------------------------------------------------------------------------------
    Resource* Computation::getResource( const std::string& handle ) 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    const Resource* Computation::getResource( const std::string& handle ) const 
    { 
        return NULL; 
    }

    //------------------------------------------------------------------------------
    void Computation::getResources( ResourceList& resourceList, const std::string& handle ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Computation::getAllResources( ResourceList& resourceList ) 
    { 
    }

    //------------------------------------------------------------------------------
    void Computation::enable() 
    { 
        _enabled = true; 
    }

    //------------------------------------------------------------------------------
    void Computation::disable() 
    { 
        _enabled = false; 
    }

    //------------------------------------------------------------------------------
    bool Computation::isEnabled() const
    {
        return _enabled;
    }

	//------------------------------------------------------------------------------
	const std::string& Computation::getLibraryName() const
	{
		return _libraryName;
	}

	//------------------------------------------------------------------------------
	std::string& Computation::getLibraryName()
	{
		return _libraryName;
	}

	//------------------------------------------------------------------------------
	void Computation::setLibraryName( const std::string& libraryName )
	{
		_libraryName = libraryName;
	}

    //------------------------------------------------------------------------------
    void Computation::clear()
    {
        clearLocal();
        Resource::clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Computation::clearLocal()
    {
        _eventCallback = NULL;
        _updateCallback = NULL;
        _enabled = true;
    }
}
