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
#include "TexStreamer"
        
extern "C"
void swap( osg::Vec2s& numBlocks, osg::Vec2s& numThreads, osg::Vec4ub* target, osg::Vec4ub* source );

extern "C"
void filter( osg::Vec2s& numBlocks, osg::Vec2s& numThreads, osg::Vec4ub* trgBuffer, cudaArray* srcArray, cudaChannelFormatDesc& srcDesc );

namespace TexDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    bool TexStreamer::init() 
    { 
        if( !_trgBuffer.valid() || !_srcArray.valid() )
        {
            osg::notify( osg::WARN ) << "TexDemo::TexStreamer::init(): params are missing."
                                     << std::endl;
            return false;
        }

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // texture size must be a multiple of 16x16 texels
        _numThreads = osg::Vec2s( 16, 16 );
        _numBlocks = osg::Vec2s( _trgBuffer->getDimension(0)/16, 
                                 _trgBuffer->getDimension(1)/16 );

        return osgCuda::Module::init();
    }

    //------------------------------------------------------------------------------  
    void TexStreamer::launch( const osgCompute::Context& context ) const
    {
        if( isDirty() )
            return;

        // map params
        cudaArray* srcArray = _srcArray->mapArray( context, osgCompute::MAP_DEVICE_SOURCE );
        cudaChannelFormatDesc srcDesc = _srcArray->getChannelFormatDesc();

        osg::Vec4ub* trgBuffer0 = _trgBuffer->map( context, osgCompute::MAP_DEVICE, 0 );
        osg::Vec4ub* trgBuffer1 = _trgBuffer->map( context, osgCompute::MAP_DEVICE_TARGET, 1 );

        // KERNEL CALL 0 
        filter(  _numBlocks, 
                 _numThreads,
                 trgBuffer0,
                 srcArray,
                 srcDesc );

        // KERNEL CALL 1 
        swap( _numBlocks, 
              _numThreads, 
              trgBuffer1, 
              trgBuffer0 );

        // unmap params
        _trgBuffer->unmap( context, 1 );
        _trgBuffer->unmap( context, 0 );
        _srcArray->unmap( context );

    }

    //------------------------------------------------------------------------------
    void TexStreamer::acceptParam( const std::string& handle, osgCompute::Param& param )
    {
        if( handle == "TRG_BUFFER" )
            _trgBuffer = dynamic_cast<osgCuda::Vec4ubBuffer*>( &param );
        if( handle == "SRC_ARRAY" )
            _srcArray = dynamic_cast<osgCuda::Vec4ubArray*>( &param );
    }

    //------------------------------------------------------------------------------
    bool TexStreamer::usesParam( const std::string& handle ) const
    {
        if( handle == "TRG_BUFFER" ||
            handle == "SRC_ARRAY" )
            return true;

        return false;
    }

    //------------------------------------------------------------------------------
    void TexStreamer::removeParam( const std::string& handle )
    {
        if( handle == "TRG_BUFFER" )
            _trgBuffer = NULL;
        if( handle == "SRC_ARRAY" )
            _srcArray = NULL;
    }

    //------------------------------------------------------------------------------
    osgCompute::Param* TexStreamer::getParam( const std::string& handle )
    {
        if( handle == "TRG_BUFFER" )
            return _trgBuffer.get();
        if( handle == "SRC_ARRAY" )
            return _srcArray.get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    const osgCompute::Param* TexStreamer::getParam( const std::string& handle ) const
    {
        if( handle == "TRG_BUFFER" )
            return _trgBuffer.get();
        if( handle == "SRC_ARRAY" )
            return _srcArray.get();

        return NULL;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void TexStreamer::clearLocal() 
    { 
        _numBlocks[0] = 1;
        _numBlocks[1] = 1;
        _numThreads[0] = 1;
        _numThreads[1] = 1;
        _trgBuffer = NULL;
        _srcArray = NULL;
    }

    //------------------------------------------------------------------------------
    bool TexStreamer::init( const osgCompute::Context& context ) const
    { 
        // If cuda resources should be directly allocated by calling e.g. "cudaMalloc()" 
        // without using Buffers, Arrays and so on then don't forget to call "osgCuda::Module::init( context );" 
        // at the end of this method in order to register the module at the current context. 
        // Or call "context.registerResource( *this );" elsewhere 
        return osgCuda::Module::init( context );
    }

    //------------------------------------------------------------------------------
    void TexStreamer::clear( const osgCompute::Context& context ) const
    {
        // If you have allocated resources ( e.g. cudaMalloc() ) then here is the place
        // where you should free them (e.g. "cudaFree()" ). 
        // If so then don't forget to call "osgCuda::Module::clear( context );" at the end in order to
        // unregister this module from the context.
        // Or call "context.unregisterResource( *this );" elsewhere 
        osgCuda::Module::clear( context );
    }
}
