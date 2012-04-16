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
void trace( unsigned int numPtcls, void* ptcls, float etime );

namespace PtclDemo
{
    class PtclTracer : public osgCompute::Computation 
    {
    public:
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

    private:
        osg::ref_ptr<osgCuda::Timer>        _timer;
        osg::ref_ptr<osgCompute::Memory>    _ptcls;
    };

    //------------------------------------------------------------------------------  
    void PtclTracer::launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "PtclTracer");
        }

        _timer->start();

        trace( 
            _ptcls->getNumElements(), 
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ), 
            0.009f );

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
