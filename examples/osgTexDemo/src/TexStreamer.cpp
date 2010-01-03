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
#include <osgCuda/Buffer>
#include "TexStreamer"
        
// Declare CUDA-kernel functions
extern "C"
void gauss( const dim3& numBlocks, const dim3& numThreads, void* target, void* source, unsigned int byteSize );

extern "C"
void sobel( const dim3& numBlocks, const dim3& numThreads, void* trgBuffer, void* srcBuffer, unsigned int byteSize );

extern "C"
void swap( const dim3& numBlocks, const dim3& numThreads, void* trgBuffer, void* srcArray );

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
        // In this case we restrict our texture size to a multiple of 16 for
		// each dimension
        _threads = dim3( 16, 16, 1 );
        _blocks = dim3( _trgBuffer->getDimension(0)/16, 
                        _trgBuffer->getDimension(1)/16, 1 );

		// Do not forget to call osgCompute::Module::init()!!!
        return osgCompute::Module::init();
    }
  
    //------------------------------------------------------------------------------  
    void TexStreamer::launch()
    {
        if( isClear() )
            return;

        // Swap RGB channels 
        swap(  _blocks, 
               _threads,
               _tmpBuffer->map(),
			   _srcArray->map( osgCompute::MAP_DEVICE_SOURCE ) );

        // Run a 3x3 sobel filter 
        sobel( _blocks, 
			   _threads, 
               _trgBuffer->map(), 
               _tmpBuffer->map(),
			   _tmpBuffer->getByteSize() );

		// ... or a 5x5 gauss filter
		//gauss( _blocks, 
		//	_threads, 
		//	_trgBuffer->map(), 
		//	_tmpBuffer->map(),
		//	_tmpBuffer->getByteSize() );


		// You can also use the map function at any time 
		// in order to copy memory from GPU to CPU and vice versa.
		// To do so use the MAPPING flags (e.g. MAP_HOST_SOURCE).
		// Each time map() is called the buffer intern checks whether
		// he has to synchronize the memory.
		// Uncomment the following line if you want to observe the
		// generated memory of the swap() kernel.


		// unsigned char* data = static_cast<unsigned char*>( _trgBuffer->map( ctx, osgCompute::MAP_HOST_SOURCE ) );
    }

    //------------------------------------------------------------------------------
    void TexStreamer::acceptResource( osgCompute::Resource& resource )
    {
		// Search for your handles. This Method is called for each resource
		// located in the subgraph of this module.
        if( resource.isAddressedByHandle( "TRG_BUFFER" ) )
            _trgBuffer = dynamic_cast<osgCompute::Buffer*>( &resource );
        if( resource.isAddressedByHandle( "SRC_ARRAY" ) )
            _srcArray = dynamic_cast<osgCompute::Buffer*>( &resource );
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
