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

#ifndef ENDIANNESS_KERNEL_H
#define ENDIANNESS_KERNEL_H 1

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
__device__
inline unsigned int thIdx()
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int width = gridDim.x * blockDim.x;

    return y*width + x;
}

//-------------------------------------------------------------------------
__device__
inline unsigned int swapBytes( unsigned int value )
{
    unsigned int res =  
        ((value & 0x000000ffU) << 24)
        | ((value & 0x0000ff00U) << 8)
        | ((value & 0x00ff0000U) >> 8)
        | ((value & 0xff000000U) >> 24);

    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
__global__ 
void k_swapEndianness( unsigned int* bytes ) 
{
    // compute thread dimension
    unsigned int trgIdx = thIdx();

    // swap endianess within buffer
    bytes[trgIdx] = swapBytes( bytes[trgIdx] );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
extern "C"
void swapEndianness( unsigned int numBlocks, unsigned int numThreads, void* bytes )
{
    dim3 blocks( numBlocks, 1, 1 );
    dim3 threads( numThreads, 1, 1 );

    // call kernel
    k_swapEndianness<<< blocks, threads >>>( reinterpret_cast<unsigned int*>(bytes) );
}


#endif // ENDIANNESS_KERNEL_H