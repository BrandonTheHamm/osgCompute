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
#include <osgCompute/Computation>
#include <osgCompute/Memory>
#include <osgCudaUtil/Timer>

//------------------------------------------------------------------------------
extern "C"
void trace( unsigned int numBlocks, unsigned int numThreads, void* ptcls, float etime, unsigned int numPtcls );

namespace PtclDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // DECLARATION ///////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    class PtclTracer : public osgCompute::Computation 
    {
    public:
        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

    private:
        osg::ref_ptr<osgCuda::Timer>        _timer;
        unsigned int                        _numBlocks;
        unsigned int                        _numThreads;
        osg::ref_ptr<osgCompute::Memory>    _ptcls;
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

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "PtclTracer");
            _timer->init();
        }

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        _numBlocks = (_ptcls->getNumElements() / 128)+1;
        _numThreads = 128;

        return osgCompute::Computation::init();
    }

    //------------------------------------------------------------------------------  
    void PtclTracer::launch()
    {
        if( !_ptcls.valid() || !_timer.valid() )
            return;

        _timer->start();

        ////////////////////
        // MOVE PARTICLES //
        ////////////////////
        trace( 
            _numBlocks, 
            _numThreads, 
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ), 
            0.009f,
            _ptcls->getNumElements() );


        _timer->stop();
    }

    //------------------------------------------------------------------------------
    void PtclTracer::acceptResource( osgCompute::Resource& resource )
    {
        // Search for the particle buffer
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }
}

//-----------------------------------------------------------------------------
// Use this function to return a new warp module to the application
extern "C" OSGCOMPUTE_COMPUTATION_EXPORT osgCompute::Computation* OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION() 
{
    return new PtclDemo::PtclTracer;
}
