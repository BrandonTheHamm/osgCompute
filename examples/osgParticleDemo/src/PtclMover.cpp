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
#include <cuda_runtime.h>
#include "PtclMover"

extern "C"
void move( unsigned int numBlocks, unsigned int numThreads, osg::Vec4f* ptcls, float etime );

namespace PtclDemo
{
    //------------------------------------------------------------------------------ 
    PtclMover::~PtclMover() 
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------  
    bool PtclMover::init() 
    { 
        if( !_particles.valid() )
        {
            osg::notify( osg::WARN ) 
                << "ParticleDemo::ParticleMover::init(): particle buffer is missing."
                << std::endl;

            return false;
        }

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
    void PtclMover::launch( const osgCompute::Context& context ) const
    {
        if( isDirty() )
            return;

        if( context.getState() == NULL || context.getState()->getFrameStamp() == NULL )
            return;

        /////////////
        // ADVANCE //
        /////////////
        if( _frameStamp == NULL )
        {
            _frameStamp = context.getState()->getFrameStamp();
            _lastTime = _frameStamp->getSimulationTime();
        }

        float etime = static_cast<float>(_frameStamp->getSimulationTime() - _lastTime);
        _lastTime = _frameStamp->getSimulationTime();

        /////////////
        // MAPPING //
        /////////////
        osg::Vec4f* ptcls = (osg::Vec4f*)_particles->map( context, osgCompute::MAP_DEVICE );

        ///////////////
        // ADVECTION //
        ///////////////

        move( _numBlocks, 
              _numThreads, 
              ptcls, 
              etime );

        //_particles->unmap( context );
    }

    //------------------------------------------------------------------------------
    void PtclMover::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isAddressedByHandle("PTCL_BUFFER") )
            _particles = dynamic_cast<osgCuda::Vec4fBuffer*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void PtclMover::clearLocal() 
    { 
        _numBlocks = 1;
        _numThreads = 1;

        _particles = NULL;

        _lastTime = 0.0;
        _frameStamp = NULL;
    }
}
