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
#include <osg/Array>
#include <osg/ref_ptr>
#include <osgCompute/Program>
#include <osgCompute/Memory>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>
#include <osgCudaUtil/Timer>

//------------------------------------------------------------------------------
extern "C"
void emit( unsigned int numPtcls,
          void* ptcls,
          void* seeds,
          unsigned int seedIdx,
          osg::Vec3f bbmin,
          osg::Vec3f bbmax );


namespace PtclDemo
{
    class PtclEmitter : public osgCompute::Program 
    {
    public:
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

    protected:
        osg::ref_ptr<osgCuda::Timer>                      _timer;
        osg::ref_ptr<osgCompute::Memory>                  _ptcls;
        osg::ref_ptr<osgCompute::Memory>                  _seeds;
    };

    //------------------------------------------------------------------------------
    void PtclEmitter::launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_seeds.valid() || !_ptcls.valid() )
        {
            osg::Image* seedValues = new osg::Image();
            seedValues->allocateImage(_ptcls->getNumElements(),1,1,GL_LUMINANCE,GL_FLOAT);

            float* seeds = (float*)seedValues->data();
            for( unsigned int s=0; s<_ptcls->getNumElements(); ++s )
                seeds[s] = ( float(rand()) / RAND_MAX );

            osg::ref_ptr<osgCuda::Buffer> seedBuffer = new osgCuda::Buffer;
            seedBuffer->setName("SEED BUFFER");
            seedBuffer->setElementSize( sizeof(float) );
            seedBuffer->setDimension(0,_ptcls->getNumElements());
            seedBuffer->setImage( seedValues );
            _seeds = seedBuffer;
        }

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "PtclEmitter");

        }

        _timer->start();

        emit(
            _ptcls->getNumElements(),
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ),
            _seeds->map( osgCompute::MAP_DEVICE_SOURCE ),
            (unsigned int)(rand()),
            osg::Vec3f(-1.f,-1.f,-1.f),
            osg::Vec3f(1.f,1.f,1.f) );

        _timer->stop();
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }
}

//-----------------------------------------------------------------------------
// Use this function to return a new warp module to the application
extern "C" OSGCOMPUTE_PROGRAM_EXPORT osgCompute::Program* OSGCOMPUTE_CREATE_PROGRAM_FUNCTION( void ) 
{
    return new PtclDemo::PtclEmitter;
}
