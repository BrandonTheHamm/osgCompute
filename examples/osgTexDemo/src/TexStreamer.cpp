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

#include "TexStreamer"
        
extern "C"
void swap( osg::Vec2s& numBlocks, osg::Vec2s& numThreads, void* target, void* source );

extern "C"
void filter( osg::Vec2s& numBlocks, osg::Vec2s& numThreads, void* trgBuffer, void* srcArray );

namespace TexDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    bool TexStreamer::init() 
    { 
        if( !_trgBuffer || !_trgTmpBuffer || !_srcArray )
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

        return osgCompute::Module::init();
    }
  
    //------------------------------------------------------------------------------  
    void TexStreamer::launch( const osgCompute::Context& context ) const
    {
        if( isClear() )
            return;

        // map params
        void* srcArray = _srcArray->map( context );
        void* trgTmpBuffer = _trgTmpBuffer->map( context );
        void* trgBuffer = _trgBuffer->map( context );

        // KERNEL CALL 0 
        filter(  _numBlocks, 
                 _numThreads,
                 trgTmpBuffer,
                 srcArray );
                 //srcDesc );

        // KERNEL CALL 1 
        swap( _numBlocks, 
              _numThreads, 
              trgBuffer, 
              trgTmpBuffer );
    }

    //------------------------------------------------------------------------------
    void TexStreamer::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isAddressedByHandle( "TRG_BUFFER" ) )
            _trgBuffer = dynamic_cast<osgCompute::Buffer*>( &resource );
        if( resource.isAddressedByHandle( "TRG_TMP_BUFFER" ) )
            _trgTmpBuffer = dynamic_cast<osgCompute::Buffer*>( &resource );
        if( resource.isAddressedByHandle( "SRC_ARRAY" ) )
            _srcArray = dynamic_cast<osgCompute::Buffer*>( &resource );
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
            return _trgBuffer;
        if( handle == "SRC_ARRAY" )
            return _srcArray; 
        if( handle == "TRG_TMP_BUFFER" )
            return _trgTmpBuffer;

        return NULL;
    }

    //------------------------------------------------------------------------------
    const osgCompute::Resource* TexStreamer::getResource( const std::string& handle ) const
    {
        if( handle == "TRG_BUFFER" )
            return _trgBuffer;
        if( handle == "SRC_ARRAY" )
            return _srcArray;
        if( handle == "TRG_TMP_BUFFER" )
            return _trgTmpBuffer;

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
}
