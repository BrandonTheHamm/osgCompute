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
#include <cuda_runtime.h>
#include "PtclMapper"

namespace PtclDemo
{
    //------------------------------------------------------------------------------ 
    PtclMapper::~PtclMapper() 
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------  
    bool PtclMapper::init() 
    { 
        if( !_particlesGL.valid() || !_particlesCUDA.valid() )
        {
            osg::notify( osg::WARN ) 
                << "PtclMapper::init(): buffer is missing."
                << std::endl;

            return false;
        }

        if( _particlesGL->isDirty() )
            _particlesGL->init();

        if( _particlesCUDA->isDirty() )
            _particlesCUDA->init();

        if( _particlesGL->getByteSize() != _particlesCUDA->getByteSize() )
        {
            osg::notify( osg::WARN ) 
                << "PtclMapper::init(): buffers differ in their size."
                << std::endl;

            return false;
        }

        return osgCuda::Module::init();
    }

    //------------------------------------------------------------------------------  
    void PtclMapper::launch( const osgCompute::Context& context ) const
    {
        if( _syncType == DO_NOT_SYNC )
            return;

        switch( _syncType )
        {
        case SYNC_CUDA_WITH_GL_DEVICE:
            {
                void* glBuffer = _particlesGL->map( context, osgCompute::MAP_DEVICE_SOURCE );
                void* cudaBuffer = _particlesCUDA->map( context, osgCompute::MAP_DEVICE_TARGET );

                cudaError res = cudaMemcpy( cudaBuffer, glBuffer, _particlesCUDA->getByteSize(), cudaMemcpyDeviceToDevice );
                if( res != cudaSuccess )
                {
                    osg::notify( osg::FATAL ) 
                        << "PtclMapper::launch(): memcpy to CUDA device buffer failed."
                        << std::endl;
                }

                //_particlesCUDA->unmap( context );
                //_particlesGL->unmap( context );
            }
            break;
        case SYNC_GL_WITH_CUDA_DEVICE:
            {

                //osg::Vec4f* cudaBuffer = _particlesCUDA->map( context, osgCompute::MAP_HOST_SOURCE );
                //_particlesCUDA->unmap( context );


                void* cudaBuffer = _particlesCUDA->map( context, osgCompute::MAP_DEVICE_SOURCE );
                void* glBuffer = _particlesGL->map( context, osgCompute::MAP_DEVICE_TARGET );

                cudaError res = cudaMemcpy( glBuffer, cudaBuffer, _particlesGL->getByteSize(), cudaMemcpyDeviceToDevice );
                if( res != cudaSuccess )
                {
                    osg::notify( osg::FATAL ) 
                        << "PtclMapper::launch(): memcpy to GL device buffer failed."
                        << std::endl;
                }

                //_particlesGL->unmap( context );
                //_particlesCUDA->unmap( context );
            }
            break;
        case SYNC_CUDA_WITH_GL_HOST:
            {
                void* glBuffer = _particlesGL->map( context, osgCompute::MAP_HOST_SOURCE );
                void* cudaBuffer = _particlesCUDA->map( context, osgCompute::MAP_HOST_TARGET );

                cudaError res = cudaMemcpy( cudaBuffer, glBuffer, _particlesCUDA->getByteSize(), cudaMemcpyHostToHost );
                if( res != cudaSuccess )
                {
                    osg::notify( osg::FATAL ) 
                        << "PtclMapper::launch(): memcpy to CUDA host buffer failed."
                        << std::endl;
                }

                //_particlesCUDA->unmap( context );
                //_particlesGL->unmap( context );
            }
            break;
        case SYNC_GL_WITH_CUDA_HOST:
            {
                void* cudaBuffer = _particlesCUDA->map( context, osgCompute::MAP_HOST_SOURCE );
                void* glBuffer = _particlesGL->map( context, osgCompute::MAP_HOST_TARGET );

                cudaError res = cudaMemcpy( glBuffer, cudaBuffer, _particlesGL->getByteSize(), cudaMemcpyHostToHost );
                if( res != cudaSuccess )
                {
                    osg::notify( osg::FATAL ) 
                        << "PtclMapper::launch(): memcpy to GL host buffer failed."
                        << std::endl;
                }

                //_particlesGL->unmap( context );
                //_particlesCUDA->unmap( context );
            }
            break;
        }
    }

    //------------------------------------------------------------------------------
    void PtclMapper::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isAddressedByHandle("PTCL_GL_BUFFER")  )
            _particlesGL = dynamic_cast<osgCuda::Vec4fGeometry*>( &resource );
        if( resource.isAddressedByHandle("PTCL_CUDA_BUFFER") )
            _particlesCUDA = dynamic_cast<osgCuda::Vec4fBuffer*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void PtclMapper::clearLocal() 
    { 
        _particlesGL = NULL;
        _particlesCUDA = NULL;
        _syncType = DO_NOT_SYNC;
    }
}
