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
#include "PtclEmitter"

//------------------------------------------------------------------------------
extern "C"
void reseed( unsigned int numBlocks,
             unsigned int numThreads,
             void* ptcls,
             void* seeds,
             unsigned int seedCount,
             unsigned int seedIdx,
             float3 bbmin,
             float3 bbmax );


namespace PtclDemo
{
    //------------------------------------------------------------------------------
    PtclEmitter::~PtclEmitter()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool PtclEmitter::init()
    {
        if( !_ptcls.valid() || !_seeds.valid() )
        {
            osg::notify( osg::WARN )
                << "ParticleDemo::ParticleMover::init(): buffers are missing."
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
    void PtclEmitter::launch( const osgCompute::Context& ctx ) const
    {
        if( isClear() )
            return;

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
            _ptcls->map( ctx ),
            _seeds->map( ctx ),
            _seeds->getDimension(0),
            seedIdx,
            bbmin,
            bbmax );
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isAddressedByHandle("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Buffer*>( &resource );
        if( resource.isAddressedByHandle("PTCL_SEEDS") )
            _seeds = dynamic_cast<osgCompute::Buffer*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PtclEmitter::clearLocal()
    {
        _numBlocks = 1;
        _numThreads = 1;
        _ptcls = NULL;
        _seeds = NULL;
        _seedBoxMin = osg::Vec3f(0,0,0);
        _seedBoxMax = osg::Vec3f(0,0,0);
    }
}
