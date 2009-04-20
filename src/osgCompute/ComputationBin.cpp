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
#include "osgCompute/ComputationBin"
#include "osgCompute/Computation"
#include "osgCompute/Context"

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ComputationBin::ComputationBin()
        : osgUtil::RenderBin()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    ComputationBin::ComputationBin( osgUtil::RenderBin::SortMode mode )
        : osgUtil::RenderBin( mode )
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void ComputationBin::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void ComputationBin::drawImplementation( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
    { 
        // render sub-graph
        osgUtil::RenderBin::drawImplementation(renderInfo, previous );

        if( _context.valid() )
        {
            // Apply context 
            _context->apply();

            // Launch Modules
            if( _launchCallback ) 
                (*_launchCallback)( *this, *_context ); 
            else launch(); 
        }

        // don't forget to decrement dynamic object count
        renderInfo.getState()->decrementDynamicObjectCount();
    }

    //------------------------------------------------------------------------------
    unsigned int ComputationBin::computeNumberOfDynamicRenderLeaves() const
    {
        // increment dynamic object count to execute modules
        return osgUtil::RenderBin::computeNumberOfDynamicRenderLeaves() + 1;
    }

    //------------------------------------------------------------------------------
    bool ComputationBin::init( Computation& computation )
    {
        // COMPUTATION 
        _computation = &computation;

        // PARAMS
        if( _computation->getParamHandles() )
            _paramHandles = *_computation->getParamHandles();

        // MODULES 
        if( _computation->hasModules())
            _modules = *_computation->getModules();

        // OBJECT 
        setName( _computation->getName() );
        setDataVariance( _computation->getDataVariance() );

        // CALLBACK 
        _launchCallback = _computation->getLaunchCallback();

        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void ComputationBin::reset()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool ComputationBin::hasModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool ComputationBin::hasModule( Module& module ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr) == &module )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    bool ComputationBin::hasModules() const 
    { 
        return !_modules.empty(); 
    }

    //-----------------------------------------------------------------------------
    Module* ComputationBin::getModule( const std::string& moduleName )
    {
        for( ModuleListItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //-----------------------------------------------------------------------------
    const Module* ComputationBin::getModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->getName() == moduleName && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    ModuleList* ComputationBin::getModules() 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    const ModuleList* ComputationBin::getModules() const 
    { 
        return &_modules; 
    }

    //------------------------------------------------------------------------------
    unsigned int ComputationBin::getNumModules() const 
    { 
        return _modules.size(); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ComputationBin::clearLocal()
    {
        _computation = NULL;
        _context = NULL;
        _dirty = true;
        _launchCallback = NULL;
        _modules.clear();
        _paramHandles.clear();

        osgUtil::RenderBin::reset();
    }

    //------------------------------------------------------------------------------
    void ComputationBin::launch() const
    {
        if( _dirty || !_computation || !_context.valid() )
            return;

        ////////////////////
        // LAUNCH MODULES //
        ////////////////////
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->isEnabled() )
            {
                if( (*itr)->getLaunchCallback() ) 
                    (*(*itr)->getLaunchCallback())( *(*itr), *_context );
                // launch module
                (*itr)->launch( *_context );
            }
        }
    }

}