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
#include "PtclMover"

//------------------------------------------------------------------------------
extern "C"
void move( unsigned int numBlocks, unsigned int numThreads, void* ptcls, float etime );

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
        if( !_ptcls )
        {
            osg::notify( osg::WARN ) 
                << "PtclDemo::PtclMover::init(): particle buffer is missing."
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
    void PtclMover::launch( const osgCompute::Context& ctx ) const
    {
        if( isClear() )
            return;

        if( !ctx.isConnectedWithGraphicsContext() )
            return;
		const osg::State* state = ctx.getGraphicsContext()->getState();
		const osg::FrameStamp* fs = state->getFrameStamp();
		if( fs == NULL )
			return;

        /////////////
        // ADVANCE //
        /////////////
		if( _firstFrame )
		{
			_lastTime = fs->getSimulationTime();
			_firstFrame = false;
		}
        float elapsedtime = static_cast<float>(fs->getSimulationTime() - _lastTime);
        _lastTime = fs->getSimulationTime();

        ////////////////////
        // MOVE PARTICLES //
        ////////////////////
        move( _numBlocks, 
              _numThreads, 
              _ptcls->map( ctx ), 
              elapsedtime );
    }

    //------------------------------------------------------------------------------
    void PtclMover::acceptResource( osgCompute::Resource& resource )
    {
		// Search for the particle buffer
        if( resource.isAddressedByHandle("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Buffer*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void PtclMover::clearLocal() 
    { 
        _numBlocks = 1;
        _numThreads = 1;

        _ptcls = NULL;

        _lastTime = 0.0;
        _firstFrame = true;
    }
}
