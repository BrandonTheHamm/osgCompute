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

#include <osg/Notify>
#include "osgCompute/ProcessorBin"
#include "osgCompute/Processor"
#include "osgCompute/Context"

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ProcessorBin::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void ProcessorBin::drawImplementation( osg::RenderInfo& ) const
    { 
        if( !isEnabled() || !_context.valid() )
            return;

        // Apply context 
        _context->apply();
        
        // Launch 
        if( _launchCallback ) 
            _launchCallback->launch( *this, *_context ); 
        else launch(); 
    }

    //------------------------------------------------------------------------------
    bool ProcessorBin::init( Processor& processor )
    {
        if( !_dirty )
        {
            osg::notify(osg::INFO) << "ProcessorBin::init(): ProcessorBin for Processor \""
                                    << processor.asObject()->getName()<<"\" is not dirty."
                                    << std::endl;
            return true;
        }

        // PROCESSOR 
        _processor = &processor;

        // MODULES 
        if( _processor->hasModules())
            _modules = *_processor->getModules();

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->isDirty() )
                (*itr)->init();

        // PARAMS
        if( _processor->getParamHandles() )
            _paramHandles = *_processor->getParamHandles();
        
        for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            if( (*itr).second->isDirty() )
                (*itr).second->init();

        // OBJECT 
        setName( _processor->asObject()->getName() );
        setDataVariance( _processor->asObject()->getDataVariance() );

        // CALLBACK 
        _launchCallback = _processor->getLaunchCallback();

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    bool ProcessorBin::hasModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool ProcessorBin::hasModule( Module& module ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr) == &module )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool ProcessorBin::hasModules() const 
    { 
        return !_modules.empty(); 
    }

    //-----------------------------------------------------------------------------
    Module* ProcessorBin::getModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //-----------------------------------------------------------------------------
    const Module* ProcessorBin::getModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    ModuleList* ProcessorBin::getModules() 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    const ModuleList* ProcessorBin::getModules() const 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    unsigned int ProcessorBin::getNumModules() const 
    { 
        return _modules.size(); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ProcessorBin::clearLocal()
    {
        osg::Drawable::setSupportsDisplayList(false); 
        osg::Object::setDataVariance( osg::Object::DYNAMIC ); 

        _processor = NULL;
        _context = NULL;
        _dirty = true;
        _launchCallback = NULL;
        _enabled = true;
        _modules.clear();
        _paramHandles.clear();
    }

    //------------------------------------------------------------------------------
    void ProcessorBin::clear( const Context& context ) const
    {
        if( _context == &context )
            _context = NULL;
    }

    //------------------------------------------------------------------------------
    void ProcessorBin::launch() const
    {
        if( _dirty || !_processor || !_context.valid() )
            return;

        ////////////////////
        // LAUNCH MODULES //
        ////////////////////
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->isEnabled() )
            {
                if( (*itr)->getLaunchCallback() ) 
                    (*itr)->getLaunchCallback()->launch( *(*itr), *_context );
                // launch module
                (*itr)->launch( *_context );
            }
        }

    }
}