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

#include <builtin_types.h>
#include <math.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include "PtclEmitter"

extern "C"
void reseed( unsigned int numBlocks,
             unsigned int numThreads,
             osg::Vec4f* ptcls,
             float* seeds,
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
        if( !_particles.valid() || !_seedBoxMax.valid() || !_seedBoxMin.valid() )
        {
            osg::notify( osg::WARN )
                << "ParticleDemo::ParticleMover::init(): params are missing."
                << std::endl;

            return false;
        }

        _seedCount = 1024;

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // One Thread handles a single particle
        // buffer size must be a multiple of 128 x sizeof(float4)
        _numBlocks = _particles->getDimension(0) / 128;
        _numThreads = 128;

        return osgCuda::Module::init();
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::launch( const osgCompute::Context& context ) const
    {
        if( isDirty() )
            return;

        if( _context == NULL )
            init( context );

        //////////
        // SEED //
        //////////
        unsigned int seedIdx = static_cast<unsigned int>(rand());

        /////////////
        // MAPPING //
        /////////////
        osg::Vec4f* ptcls = _particles->map( context, osgCompute::MAP_DEVICE, 0 );
        osg::Vec3f* seedBoxMin = _seedBoxMin->data( context );
        osg::Vec3f* seedBoxMax = _seedBoxMax->data( context );

        ///////////////
        // RESEEDING //
        ///////////////

        reseed(
            _numBlocks,
            _numThreads,
            ptcls,
            _seeds,
            _seedCount,
            seedIdx,
            *seedBoxMin,
            *seedBoxMax );

        _particles->unmap( context );
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::acceptParam( const std::string& handle, osgCompute::Param& param )
    {
        if( handle == "PTCL_BUFFER" )
            _particles = dynamic_cast<osgCuda::Vec4fBuffer*>( &param );
        else if( handle == "PTCL_SEED_BOX_MIN" )
            _seedBoxMin = dynamic_cast<osgCuda::Vec3fConstant*>( &param );
        else if( handle == "PTCL_SEED_BOX_MAX" )
            _seedBoxMax = dynamic_cast<osgCuda::Vec3fConstant*>( &param );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PtclEmitter::clearLocal()
    {
        _numBlocks = 1;
        _numThreads = 1;

        _particles = NULL;
        _seedBoxMin = NULL;
        _seedBoxMax = NULL;

        // clear seeds
        if( _context != NULL && _seeds != NULL )
            _context->freeMemory( _seeds );

        _seeds = NULL;
        _context = NULL;
    }

    //------------------------------------------------------------------------------
    bool PtclEmitter::init( const osgCompute::Context& context ) const
    {
        if( context.getState() == NULL )
            return false;

        ////////////////////
        // GENERATE SEEDS //
        ////////////////////
        float maxf = 1.0f;
        float minf = 0.0f;
        float* tmpPtr = new float[_seedCount];
        for( unsigned int s=0; s<_seedCount; ++s )
        {
            float unit = float(rand()) / RAND_MAX;
            float diff = maxf - minf;
            tmpPtr[s] = minf + unit * diff;
        }

        ///////////////////
        // DEVICE BUFFER //
        ///////////////////
        unsigned int bufferSize = _seedCount * sizeof(float);
        // allocate buffer on the device to store the seeds
        _context = dynamic_cast<const osgCuda::Context*>( &context );
        if( _context == NULL )
            return false;

        _seeds = static_cast<float*>( _context->mallocDeviceMemory( bufferSize ) );
        if( _seeds == NULL )
        {
            _context = NULL;
            delete [] tmpPtr;
            return false;
        }

        cudaError res = cudaMemcpy( _seeds, tmpPtr, bufferSize, cudaMemcpyHostToDevice );
        if( res != cudaSuccess )
        {
            _context->freeMemory( _seeds );
            _context = NULL;
            delete [] tmpPtr;
            return false;
        }

        // delete tmpPtr
        delete [] tmpPtr;

        if( !osgCuda::Module::init( context ) )
        {
            _context->freeMemory( _seeds );
            _context = NULL;
            return false;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::clear( const osgCompute::Context& context ) const
    {
        if( _context != NULL && _seeds != NULL )
        {
            _context->freeMemory( _seeds );
            _seeds = NULL;
        }

        osgCuda::Module::clear( context );
    }
}
