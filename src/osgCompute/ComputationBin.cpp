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
#include <osgCompute/ComputationBin>
#include <osgCompute/Computation>

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
        osg::State& state = *renderInfo.getState();
        
        // Activate Resource Entries
        Resource::setCurrentIdx( state.getContextID() );

        unsigned int numToPop = (previous ? osgUtil::StateGraph::numToPop(previous->_parent) : 0);
        if (numToPop>1) --numToPop;
        unsigned int insertStateSetPosition = state.getStateSetStackSize() - numToPop;

        if (_stateset.valid())
        {
            state.insertStateSet(insertStateSetPosition, _stateset.get());
        }

        ///////////////////
        // DRAW PRE BINS //
        ///////////////////
        osgUtil::RenderBin::RenderBinList::iterator rbitr;
        for(rbitr = _bins.begin();
            rbitr!=_bins.end() && rbitr->first<0;
            ++rbitr)
        {
            rbitr->second->draw(renderInfo,previous);
        }


        if( (_computeOrder & OSGCOMPUTE_PRE_TRAVERSAL ) == OSGCOMPUTE_PRE_TRAVERSAL )
        {
            if( _launchCallback ) 
                (*_launchCallback)( *_computation ); 
            else launch(); 

            // don't forget to decrement dynamic object count
            renderInfo.getState()->decrementDynamicObjectCount();
        }

        // render sub-graph leafs
        drawLeafs(renderInfo, previous );

        if( (_computeOrder & OSGCOMPUTE_PRE_TRAVERSAL ) != OSGCOMPUTE_PRE_TRAVERSAL )
        {
            if( _launchCallback ) 
                (*_launchCallback)( *_computation ); 
            else launch();  

            // don't forget to decrement dynamic object count
            renderInfo.getState()->decrementDynamicObjectCount();
        }

        ////////////////////
        // DRAW POST BINS //
        ////////////////////
        for(;
            rbitr!=_bins.end();
            ++rbitr)
        {
            rbitr->second->draw(renderInfo,previous);
        }


        if (_stateset.valid())
        {
            state.removeStateSet(insertStateSetPosition);
        }
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
        _computeOrder = computation.getComputeOrder();

        // MODULES 
        if( _computation->hasModules())
        {
            _modules = _computation->getModules();
        }

        // OBJECT 
        setName( _computation->getName() );
        setDataVariance( _computation->getDataVariance() );

        // CALLBACK 
        _launchCallback = _computation->getLaunchCallback();

        _clear = false;
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
            if( (*itr)->isAddressedByIdentifier(moduleName)  )
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
            if( (*itr)->isAddressedByIdentifier(moduleName) && (*itr).valid() )
                return (*itr).get();

        return NULL;
    }

    //-----------------------------------------------------------------------------
    const Module* ComputationBin::getModule( const std::string& moduleName ) const
    {
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
            if( (*itr)->isAddressedByIdentifier(moduleName) && (*itr).valid() )
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

    //------------------------------------------------------------------------------
    bool ComputationBin::isClear() const
    { 
        return _clear; 
    }

    //------------------------------------------------------------------------------
    Computation* ComputationBin::getComputation()
    { 
        return _computation; 
    }

    //------------------------------------------------------------------------------
    const Computation* ComputationBin::getComputation() const
    { 
        return _computation; 
    }

    //------------------------------------------------------------------------------
    LaunchCallback* ComputationBin::getLaunchCallback()
    { 
        return _launchCallback; 
    }

    //------------------------------------------------------------------------------
    const LaunchCallback* ComputationBin::getLaunchCallback() const
    { 
        return _launchCallback; 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void ComputationBin::clearLocal()
    {
        _computation = NULL;
        _computeOrder = osgCompute::Computation::RENDER_PRE_RENDER_PRE_TRAVERSAL;
        _clear = true;
        _launchCallback = NULL;
        _modules.clear();

        osgUtil::RenderBin::reset();
    }

    //------------------------------------------------------------------------------
    void ComputationBin::launch()
    {
        if( _clear || !_computation  )
            return;

        // Launch modules
        for( ModuleListCnstItr itr = _modules.begin(); itr != _modules.end(); ++itr )
        {
            if( (*itr)->isEnabled() )
            {
                if( (*itr)->isClear() )
                    (*itr)->init();

                (*itr)->launch();
            }
        }
    }

    //------------------------------------------------------------------------------
    void ComputationBin::drawLeafs( osg::RenderInfo& renderInfo, osgUtil::RenderLeaf*& previous )
    {
        // draw fine grained ordering.
        for(osgUtil::RenderBin::RenderLeafList::iterator rlitr= _renderLeafList.begin();
            rlitr!= _renderLeafList.end();
            ++rlitr)
        {
            osgUtil::RenderLeaf* rl = *rlitr;
            rl->render(renderInfo,previous);
            previous = rl;
        }

        // draw coarse grained ordering.
        for(osgUtil::RenderBin::StateGraphList::iterator oitr=_stateGraphList.begin();
            oitr!=_stateGraphList.end();
            ++oitr)
        {

            for(osgUtil::StateGraph::LeafList::iterator dw_itr = (*oitr)->_leaves.begin();
                dw_itr != (*oitr)->_leaves.end();
                ++dw_itr)
            {
                osgUtil::RenderLeaf* rl = dw_itr->get();
                rl->render(renderInfo,previous);
                previous = rl;

            }
        }
    }

    osgUtil::RegisterRenderBinProxy registerComputationBinProxy("osgCompute::ComputationBin", new osgCompute::ComputationBin );
}