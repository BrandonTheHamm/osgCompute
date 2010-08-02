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

#include <vector_types.h>
#include <math.h>
#include <cstdlib>
#include <osg/Notify>
#include <osg/FrameStamp>
#include <osgCompute/Module>
#include <osgCompute/Memory>

//------------------------------------------------------------------------------
extern "C"
void trace( unsigned int numBlocks, unsigned int numThreads, void* ptcls, float etime );

namespace PtclDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // DECLARATION ///////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    class PtclTracer : public osgCompute::Module 
    {
    public:
        PtclTracer() : osgCompute::Module() {clearLocal();}

        META_Object( PtclDemo, PtclTracer )

        // Modules have to implement at least this
        // three methods:
        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

        virtual void clear() { clearLocal(); osgCompute::Module::clear(); }
    protected:
        virtual ~PtclTracer() {clearLocal();}
        void clearLocal();

        double                              _lastTime;
        bool						        _firstFrame;

        unsigned int                        _numBlocks;
        unsigned int                        _numThreads;

        osg::ref_ptr<osgCompute::Memory>    _ptcls;

    private:
        PtclTracer(const PtclTracer&, const osg::CopyOp& ) {} 
        inline PtclTracer &operator=(const PtclTracer &) { return *this; }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    bool PtclTracer::init() 
    { 
        if( !_ptcls )
        {
            osg::notify( osg::WARN ) 
                << "PtclDemo::PtclTracer::init(): particle buffer is missing."
                << std::endl;

            return false;
        }

        osg::FrameStamp* fs = (osg::FrameStamp*) getUserData();
        if( !fs )
        {
            osg::notify( osg::WARN )
                << "PtclDemo::PtclTracer::init(): frame stamp is missing."
                << std::endl;

            return false;
        }

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // One Thread handles a single particle
        // buffer size must be a multiple of 128 x sizeof(float4)
        _numBlocks = _ptcls->getDimension(0) / 128;
        _numThreads = 128;

        return osgCompute::Module::init();
    }

    //------------------------------------------------------------------------------  
    void PtclTracer::launch()
    {
        if( isClear() )
            return;

        /////////////
        // ADVANCE //
        /////////////
        float time = (float)((osg::FrameStamp*) getUserData())->getSimulationTime();
        if( _firstFrame )
        {
            _lastTime = time;
            _firstFrame = false;
        }

        float elapsedtime = static_cast<float>(time - _lastTime);
        _lastTime = time;

        ////////////////////
        // MOVE PARTICLES //
        ////////////////////
        trace( _numBlocks, _numThreads, _ptcls->map(), elapsedtime );
    }

    //------------------------------------------------------------------------------
    void PtclTracer::acceptResource( osgCompute::Resource& resource )
    {
        // Search for the particle buffer
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void PtclTracer::clearLocal() 
    { 
        _numBlocks = 1;
        _numThreads = 1;

        _ptcls = NULL;

        _lastTime = 0.0;
        _firstFrame = true;
    }
}

//-----------------------------------------------------------------------------
// Use this function to return a new warp module to the application
extern "C" OSGCOMPUTE_MODULE_EXPORT osgCompute::Module* OSGCOMPUTE_CREATE_MODULE_FUNCTION() 
{
    return new PtclDemo::PtclTracer;
}
