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
#include "osgCompute/PipelineBin"
#include "osgCompute/Pipeline"
#include "osgCompute/Context"

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PipelineBin::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void PipelineBin::drawImplementation( osg::RenderInfo& ) const
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
    bool PipelineBin::init( Pipeline& pipeline )
    {
        if( !_dirty )
        {
            osg::notify(osg::INFO) << "PipelineBin::init(): PipelineBin for Pipeline \""
                                    << pipeline.asObject()->getName()<<"\" is not dirty."
                                    << std::endl;
            return true;
        }

        // PIPELINE 
        _pipeline = &pipeline;

        // MODULES 
        if( _pipeline->hasModules())
            _modules = *_pipeline->getModules();

        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->isDirty() )
                (*itr)->init();

        // PARAMS
        if( _pipeline->getParamHandles() )
            _paramHandles = *_pipeline->getParamHandles();
        
        for( HandleToParamMapItr itr = _paramHandles.begin(); itr != _paramHandles.end(); ++itr )
            if( (*itr).second->isDirty() )
                (*itr).second->init();

        // OBJECT 
        setName( _pipeline->asObject()->getName() );
        setDataVariance( _pipeline->asObject()->getDataVariance() );

        // CALLBACK 
        _launchCallback = _pipeline->getLaunchCallback();

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    bool PipelineBin::hasModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool PipelineBin::hasModule( Module& module ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr) == &module )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool PipelineBin::hasModules() const 
    { 
        return !_modules.empty(); 
    }

    //-----------------------------------------------------------------------------
    Module* PipelineBin::getModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //-----------------------------------------------------------------------------
    const Module* PipelineBin::getModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    ModuleList* PipelineBin::getModules() 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    const ModuleList* PipelineBin::getModules() const 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    unsigned int PipelineBin::getNumModules() const 
    { 
        return _modules.size(); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PipelineBin::clearLocal()
    {
        osg::Drawable::setSupportsDisplayList(false); 
        osg::Object::setDataVariance( osg::Object::DYNAMIC ); 

        _pipeline = NULL;
        _context = NULL;
        _dirty = true;
        _launchCallback = NULL;
        _enabled = true;
        _modules.clear();
        _paramHandles.clear();
    }

    //------------------------------------------------------------------------------
    void PipelineBin::clear( const Context& context ) const
    {
        if( _context == &context )
            _context = NULL;
    }

    //------------------------------------------------------------------------------
    void PipelineBin::launch() const
    {
        if( _dirty || !_pipeline || !_context.valid() )
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