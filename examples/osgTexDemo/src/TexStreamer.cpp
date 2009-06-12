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
        if( !_trgBuffer.valid() || !_trgTmpBuffer.valid() || !_srcArray.valid() )
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

        osg::Vec4ub* trgTmpBuffer = (osg::Vec4ub*)_trgTmpBuffer->map( context, osgCompute::MAP_DEVICE );
        osg::Vec4ub* trgBuffer = (osg::Vec4ub*)_trgBuffer->map( context, osgCompute::MAP_DEVICE_TARGET );

        // KERNEL CALL 0 
        filter(  _numBlocks, 
                 _numThreads,
                 trgTmpBuffer,
                 srcArray,
                 srcDesc );

        // KERNEL CALL 1 
        swap( _numBlocks, 
              _numThreads, 
              trgBuffer, 
              trgTmpBuffer );

        // unmap params
        //_trgBuffer->unmap( context );
        //_trgTmpBuffer->unmap( context );
        //_srcArray->unmap( context );

    }

    //------------------------------------------------------------------------------
    void TexStreamer::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isAddressedByHandle( "TRG_BUFFER" ) )
            _trgBuffer = dynamic_cast<osgCuda::Vec4ubTexture2D*>( &resource );
        if( resource.isAddressedByHandle( "TRG_TMP_BUFFER" ) )
            _trgTmpBuffer = dynamic_cast<osgCuda::Vec4ubBuffer*>( &resource );
        if( resource.isAddressedByHandle( "SRC_ARRAY" ) )
            _srcArray = dynamic_cast<osgCuda::Vec4ubArray*>( &resource );
    }

    //------------------------------------------------------------------------------
    bool TexStreamer::usesResource( const std::string& handle ) const
    {
        if( handle == "TRG_BUFFER" ||
            handle == "SRC_ARRAY" ||
            handle == "TRG_TMP_BUFFER" )
            return true;

        return false;
    }

    //------------------------------------------------------------------------------
    void TexStreamer::removeResource( const std::string& handle )
    {
        if( handle == "TRG_BUFFER" )
            _trgBuffer = NULL;
        if( handle == "SRC_ARRAY" )
            _srcArray = NULL;
        if( handle == "TRG_TMP_BUFFER" )
            _trgTmpBuffer = NULL;
    }

    //------------------------------------------------------------------------------
    osgCompute::Resource* TexStreamer::getResource( const std::string& handle )
    {
        if( handle == "TRG_BUFFER" )
            return _trgBuffer.get();
        if( handle == "SRC_ARRAY" )
            return _srcArray.get(); 
        if( handle == "TRG_TMP_BUFFER" )
            return _trgTmpBuffer.get();

        return NULL;
    }

    //------------------------------------------------------------------------------
    const osgCompute::Resource* TexStreamer::getResource( const std::string& handle ) const
    {
        if( handle == "TRG_BUFFER" )
            return _trgBuffer.get();
        if( handle == "SRC_ARRAY" )
            return _srcArray.get();
        if( handle == "TRG_TMP_BUFFER" )
            return _trgTmpBuffer.get();

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
        _trgTmpBuffer = NULL;
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
