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

//#include <builtin_types.h>
#include <math.h>
#include <cstdlib>
//#include <cuda_runtime.h>
#include "PtclEmitter"

extern "C"
void reseed( unsigned int numBlocks,
             unsigned int numThreads,
             void* ptcls,
             void* seeds,
             unsigned int seedCount,
             unsigned int seedIdx,
             osg::Vec3f bbmin,
             osg::Vec3f bbmax );


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
        if( !_ptcls || !_seeds )
        {
            osg::notify( osg::WARN )
                << "ParticleDemo::ParticleMover::init(): params are missing."
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
    void PtclEmitter::launch( const osgCompute::Context& context ) const
    {
        if( isDirty() )
            return;

        ////////////
        // PARAMS //
        ////////////
        unsigned int seedIdx = static_cast<unsigned int>(rand());
        void* ptcls = _ptcls->map( context, osgCompute::MAP_DEVICE );
        void* seeds = _seeds->map( context, osgCompute::MAP_DEVICE_SOURCE );

        ///////////////
        // RESEEDING //
        ///////////////
        reseed(
            _numBlocks,
            _numThreads,
            ptcls,
            seeds,
            _seeds->getDimension(0),
            seedIdx,
            _seedBoxMin,
            _seedBoxMax );
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

    ////------------------------------------------------------------------------------
    //bool PtclEmitter::init( const osgCompute::Context& context ) const
    //{
    //    if( context.getState() == NULL )
    //        return false;

    //    ////////////////////
    //    // GENERATE SEEDS //
    //    ////////////////////
    //    //float maxf = 1.0f;
    //    //float minf = 0.0f;
    //    float* tmpPtr = new float[_seedCount];
    //    for( unsigned int s=0; s<_seedCount; ++s )
    //    {
    //        //float unit = float(rand()) / RAND_MAX;
    //        //float diff = maxf - minf;
    //        tmpPtr[s] = frand(0.0f,1.0f);//minf + unit * diff;
    //    }

    //    ///////////////////
    //    // DEVICE BUFFER //
    //    ///////////////////
    //    unsigned int bufferSize = _seedCount * sizeof(float);
    //    // allocate buffer on the device to store the seeds
    //    _context = dynamic_cast<const osgCompute::Context*>( &context );
    //    if( _context == NULL )
    //        return false;

    //    _seeds = static_cast<float*>( _context->mallocDeviceMemory( bufferSize ) );
    //    if( _seeds == NULL )
    //    {
    //        _context = NULL;
    //        delete [] tmpPtr;
    //        return false;
    //    }

    //    cudaError res = cudaMemcpy( _seeds, tmpPtr, bufferSize, cudaMemcpyHostToDevice );
    //    if( res != cudaSuccess )
    //    {
    //        _context->freeMemory( _seeds );
    //        _context = NULL;
    //        delete [] tmpPtr;
    //        return false;
    //    }

    //    // delete tmpPtr
    //    delete [] tmpPtr;

    //    if( !osgCompute::Module::init( context ) )
    //    {
    //        _context->freeMemory( _seeds );
    //        _context = NULL;
    //        return false;
    //    }

    //    ////////////////
    //    // INIT PTCLS //
    //    ////////////////
    //    osg::Vec4f* ptcls = (osg::Vec4f*)_particles->map( context, osgCompute::MAP_HOST_TARGET );

    //    for( unsigned int v=0; v<_particles->getDimension(0); ++v )
    //        ptcls[v].set(-1,-1,-1,1);

    //    _particles->unmap( context );

    //    return true;
    //}

    ////------------------------------------------------------------------------------
    //void PtclEmitter::clear( const osgCompute::Context& context ) const
    //{
    //    if( _context != NULL && _seeds != NULL )
    //    {
    //        dynamic_cast<osgCompute::Context>_context->freeMemory( _seeds );
    //        _seeds = NULL;
    //    }

    //    osgCompute::Module::clear( context );
    //}
}
