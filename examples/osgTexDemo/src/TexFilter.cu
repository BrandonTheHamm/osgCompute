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
texture<uchar4, 2, cudaReadModeNormalizedFloat> swapTex; 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
__global__ 
void swapKernel( unsigned int imageWidth, unsigned int imageHeight, float4* trg, unsigned int trgPitch ) 
{
    // compute thread dimension
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x < imageWidth && y < imageHeight )
    {

        float offset_X = (1.f/(float)imageWidth)*0.5f;
        float offset_Y = (1.f/(float)imageHeight)*0.5f;
        // compute texture coordinates
        float2 texCoord = make_float2( ((float) x / (float) imageWidth)+offset_X ,
                                       ((float) y / (float) imageHeight)+offset_Y );
        
        // sample value
        float4 src = tex2D( swapTex, texCoord.x, texCoord.y );
       
        // compute target address 
        float4* target = (float4*)(((char*) trg) + trgPitch * y ) + x;

        // swap channels
        (*target) = make_float4(src.z,src.x,src.y,src.w);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
extern "C"
void swap( unsigned int imageWidth, unsigned int imageHeight, void* srcArray, void* trgBuffer, unsigned int trgPitch )
{
    dim3 blocks = dim3(imageWidth / 16, imageHeight / 16, 1 ); 
    dim3 threads = dim3( 16, 16, 1 );

    if( imageWidth % 16 != 0 )
        blocks.x++;
    if( imageHeight % 16 != 0 )
        blocks.y++;


    // set texture parameters
    swapTex.normalized = true;                      // normalized texture coordinates (element of [0:1])
    swapTex.filterMode = cudaFilterModeLinear;      // bilinear interpolation 
    swapTex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    swapTex.addressMode[1] = cudaAddressModeClamp;

    // bind texture
    cudaBindTextureToArray( swapTex, reinterpret_cast<cudaArray*>(srcArray) );

    // call kernel
    swapKernel<<< blocks, threads >>>( imageWidth, imageHeight, reinterpret_cast<float4*>(trgBuffer), trgPitch );
}
