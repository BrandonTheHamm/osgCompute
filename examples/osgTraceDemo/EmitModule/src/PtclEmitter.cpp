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
#include <osgCompute/Computation>
#include <osgCompute/Memory>
#include <osgCuda/Memory>
#include <osgCuda/Geometry>
#include <osgCudaUtil/Timer>

//------------------------------------------------------------------------------
extern "C"
void reseed( unsigned int numBlocks,
            unsigned int numThreads,
            void* ptcls,
            void* seeds,
            unsigned int seedCount,
            unsigned int seedIdx,
            float3 bbmin,
            float3 bbmax,
            unsigned int numPtcls );


namespace PtclDemo
{
    /**
    */
    class PtclEmitter : public osgCompute::Computation 
    {
    public:
        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

    protected:
        osg::ref_ptr<osgCuda::Timer>                      _timer;
        unsigned int                                      _numBlocks;
        unsigned int                                      _numThreads;
        osg::Vec3f                                        _seedBoxMin;
        osg::Vec3f                                        _seedBoxMax;
        osg::ref_ptr<osgCompute::Memory>                  _ptcls;
        osg::ref_ptr<osgCompute::Memory>                  _seeds;
    };

    //------------------------------------------------------------------------------
    bool PtclEmitter::init()
    {
        if( !_ptcls.valid() )
        {
            osg::notify( osg::WARN )
                << "ParticleDemo::ParticleMover::init(): buffers are missing."
                << std::endl;

            return false;
        }

        _seedBoxMin = osg::Vec3f(-1.f,-1.f,-1.f);
        _seedBoxMax = osg::Vec3f(1.f,1.f,1.f);

        ////////////////////////
        // CREATE SEED BUFFER //
        ////////////////////////
		osg::Image* seedValues = new osg::Image();
		seedValues->allocateImage(64000,1,1,GL_LUMINANCE,GL_FLOAT);

		float* seeds = (float*)seedValues->data();
		for( unsigned int s=0; s<64000; ++s )
			seeds[s] = ( float(rand()) / RAND_MAX );

        osg::ref_ptr<osgCuda::Memory> seedBuffer = new osgCuda::Memory;
        seedBuffer->setElementSize( sizeof(float) );
        seedBuffer->setDimension(0,_ptcls->getNumElements());
        seedBuffer->setImage( seedValues );
        _seeds = seedBuffer;

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // One Thread handles a single particle
        // buffer size must be a multiple of 128 x sizeof(float4)
        _numBlocks = (_ptcls->getDimension(0) / 128)+1;
        _numThreads = 128;


        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "PtclEmitter");
            _timer->init();
        }


        return osgCompute::Computation::init();
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::launch()
    {
        if( !_seeds.valid() || !_ptcls.valid() || !_timer.valid() )
            return;

        _timer->start();

        ////////////
        // PARAMS //
        ////////////
        unsigned int seedIdx = static_cast<unsigned int>(rand());
        float3 bbmin;
        bbmin.x = _seedBoxMin.x();
        bbmin.y = _seedBoxMin.y();
        bbmin.z = _seedBoxMin.z();

        float3 bbmax;
        bbmax.x = _seedBoxMax.x();
        bbmax.y = _seedBoxMax.y();
        bbmax.z = _seedBoxMax.z();

        ////////////
        // RESEED //
        ////////////
        reseed(
            _numBlocks,
            _numThreads,
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ),
            _seeds->map( osgCompute::MAP_DEVICE_SOURCE ),
            _seeds->getDimension(0),
            seedIdx,
            bbmin,
            bbmax,
            _ptcls->getNumElements());

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
extern "C" OSGCOMPUTE_COMPUTATION_EXPORT osgCompute::Computation* OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION( void ) 
{
    return new PtclDemo::PtclEmitter;
}
