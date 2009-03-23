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

#ifndef TEXDEMO_TEXSTREAMER_KERNEL_H
#define TEXDEMO_TEXSTREAMER_KERNEL_H 1

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
inline unsigned int thIdx()
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int width = gridDim.x * blockDim.x;

    return y*width + x;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
__global__ 
void k_swap( uchar4* trg, uchar4* src ) 
{
    // compute thread dimension
    unsigned int trgIdx = thIdx();

    // swap pixel within buffer
    trg[trgIdx] = make_uchar4( 
        src[trgIdx].z, 
        src[trgIdx].y, 
        src[trgIdx].x,  
        src[trgIdx].w );
}

texture<uchar4, 2, cudaReadModeNormalizedFloat> srcTex; 
 
//-------------------------------------------------------------------------
__global__ 
void k_filter( uchar4* trg ) 
{
    // compute thread dimension
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int width = gridDim.x * blockDim.x;
    unsigned int height = gridDim.y * blockDim.y;

    // compute target idx
    unsigned int trgIdx = y*width + x;

    // compute texture coordinates
    float2 texCoord = make_float2( ((float) x / (float) width) ,
                                   ((float) y / (float) height) );

    // sample value
    float4 src = tex2D( srcTex, texCoord.x, texCoord.y );
    // write to output
    trg[trgIdx] = make_uchar4( 
                        (unsigned char)(src.x*255.0f), 
                        (unsigned char)(src.y*255.0f),
                        (unsigned char)(src.z*255.0f),
                        (unsigned char)(src.w*255.0f));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <osg/Vec4f>
#include <osg/Vec4ub>
#include <osg/Vec2s>

extern "C"
void swap( osg::Vec2s& numBlocks, osg::Vec2s& numThreads, osg::Vec4ub* trg, osg::Vec4ub* src )
{
    dim3 blocks( numBlocks[0], numBlocks[1], 1 );
    dim3 threads( numThreads[0], numThreads[1], 1 );

    // call kernel
    k_swap<<< blocks, threads >>>( reinterpret_cast<uchar4*>(trg),
                                   reinterpret_cast<uchar4*>(src) );
}

extern "C"
void filter( osg::Vec2s& numBlocks, osg::Vec2s& numThreads, osg::Vec4ub* trgBuffer, cudaArray* srcArray, cudaChannelFormatDesc& srcDesc )
{
    dim3 blocks( numBlocks[0], numBlocks[1], 1 );
    dim3 threads( numThreads[0], numThreads[1], 1 );

    // set texture parameters
    srcTex.normalized = true;                      // normalized texture coordinates (element of [0:1])
    srcTex.filterMode = cudaFilterModeLinear;      // bilinear interpolation 
    srcTex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    srcTex.addressMode[1] = cudaAddressModeClamp;

    // bind texture
    cudaError res = cudaBindTextureToArray( srcTex, srcArray, srcDesc );

    // call kernel
    k_filter<<< blocks, threads >>>( reinterpret_cast<uchar4*>(trgBuffer) );
}

#endif // TEXDEMO_TEXSTREAMER_KERNEL_H