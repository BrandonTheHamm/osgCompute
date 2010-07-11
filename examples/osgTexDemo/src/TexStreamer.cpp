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
#include <cuda_runtime.h>
#include <osg/Notify>
#include <osgCuda/Buffer>
#include <osgCuda/Texture>
#include "TexStreamer"

// Declare CUDA-kernel functions
extern "C"
void gauss( const dim3& numBlocks, const dim3& numThreads, void* target, void* source, unsigned int byteSize );

extern "C"
void sobel( const dim3& numBlocks, const dim3& numThreads, void* trgBuffer, void* srcBuffer, unsigned int byteSize );

extern "C"
void swap( const dim3& blocks, const dim3& threads, void* trgBuffer, void* srcArray, unsigned int trgPitch, unsigned int imageWidth, unsigned int imageHeight );

namespace TexDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    bool TexStreamer::init() 
    { 
        if( !_trgBuffer || !_srcArray )
        {
            osg::notify( osg::WARN ) 
                << "TexDemo::TexStreamer::init(): buffers are missing."
                << std::endl;
            return false;
        }

        // Create an internal buffer
        _tmpBuffer = new osgCuda::Buffer;
        _tmpBuffer->setElementSize( sizeof(osg::Vec4ub) );
        _tmpBuffer->setName( "trgTmpBuffer" );
        _tmpBuffer->setDimension( 0, _srcArray->getDimension(0) );
        _tmpBuffer->setDimension( 1, _srcArray->getDimension(1) );
        if( !_tmpBuffer->init() )
        {
            osg::notify( osg::WARN ) 
                << "TexDemo::TexStreamer::init(): cannot allocate temporary buffer."
                << std::endl;
            return false;
        }

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // In this case we restrict our thread grid to 256 threads per block
        _threads = dim3( 16, 16, 1 );

        unsigned int numReqBlocksWidth = 0, numReqBlocksHeight = 0;
        if( _trgBuffer->getDimension(0) % 16 == 0) 
            numReqBlocksWidth = _trgBuffer->getDimension(0) / 16;
        else
            numReqBlocksWidth = _trgBuffer->getDimension(1) / 16 + 1;

        if( _trgBuffer->getDimension(1) % 16 == 0) 
            numReqBlocksHeight = _trgBuffer->getDimension(1) / 16;
        else
            numReqBlocksHeight = _trgBuffer->getDimension(1) / 16 + 1;
        
        _blocks = dim3( numReqBlocksWidth, numReqBlocksHeight, 1 );

        // Do not forget to call osgCompute::Module::init()!!!
        return osgCompute::Module::init();
    }

    //------------------------------------------------------------------------------  
    void TexStreamer::launch()
    {
        if( isClear() )
            return;

        // Swap RGB channels 
        //swap(  _blocks, 
        //       _threads,
        //       _tmpBuffer->map(),
        //       _srcArray->map() );

        //// Run a 3x3 sobel filter 
        //sobel(_blocks, 
        //    _threads, 
        //    _trgBuffer->map(), 
        //    _tmpBuffer->map(),
        //    _tmpBuffer->getByteSize() );

        // ... or a 5x5 gauss filter
        //gauss( _blocks, 
        //	_threads, 
        //	_trgBuffer->map(), 
        //	_tmpBuffer->map(),
        //	_tmpBuffer->getByteSize() );



        swap(  _blocks, 
            _threads,
            _trgBuffer->map(),
            _srcArray->map( osgCompute::MAP_DEVICE_ARRAY ),
            _trgBuffer->getPitch(),
            _trgBuffer->getDimension(0),
            _trgBuffer->getDimension(1) );

        // You can also use the map function at any time 
        // in order to copy memory from GPU to CPU and vice versa.
        // To do so use the MAPPING flags (e.g. MAP_HOST_SOURCE).
        // Each time map() is called the buffer intern checks whether
        // he has to synchronize the memory.
        // Uncomment the following line if you want to observe the
        // generated memory of the kernel.
        // unsigned char* data = static_cast<unsigned char*>( _trgBuffer->map( osgCompute::MAP_HOST_SOURCE ) );

        // Uncomment the following line if you want to change the
        // generated memory of the kernel on the CPU. 
        //unsigned char* data = static_cast<unsigned char*>( _trgBuffer->map( osgCompute::MAP_HOST_TARGET ) );
        //for( unsigned int t=0; t<_trgBuffer->getNumElements()*4; t+=4 )
        //{
        //    data[t] = 0;
        //    data[t+1] = 255;
        //    data[t+2] = 0;
        //    data[t+3] = 0;
        //}
    }

    //------------------------------------------------------------------------------
    void TexStreamer::acceptResource( osgCompute::Resource& resource )
    {
        // Search for your handles. This Method is called for each resource
        // located in the subgraph of this module.
        if( resource.isIdentifiedBy( "TRG_BUFFER" ) )
            _trgBuffer = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy( "SRC_ARRAY" ) )
            _srcArray = dynamic_cast<osgCompute::Memory*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void TexStreamer::clearLocal() 
    { 
        _threads = dim3(0,0,0);
        _blocks = dim3(0,0,0);
        _trgBuffer = NULL;
        _tmpBuffer = NULL;
        _srcArray = NULL;
    }
}
