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


texture<uchar4, 1, cudaReadModeElementType> gaussTex; 
texture<uchar4, 2, cudaReadModeNormalizedFloat> swapTex; 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
__device__
float clamp( float value, float minValue, float maxValue )
{
    float erg = value;

    if( erg > maxValue )
        erg = maxValue;
    if( erg < minValue )
        erg = minValue;

    return erg;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
__global__ 
void gaussKernel( uchar4* trg ) 
{
    // compute thread pos
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    int xPrev2 = ((x-2) < 0)? ((gridDim.x) * blockDim.x)-1 : x-2;
    int xPrev = ((x-1) < 0)? ((gridDim.x) * blockDim.x)-1 : x-1;
    int xNext = ((x+1) >= (gridDim.x * blockDim.x))? 0 : x+1;
    int xNext2 = ((x+2) >= (gridDim.x * blockDim.x))? 0 : x+2;

    int yPrev2 = ((y-2) < 0)? (gridDim.y * blockDim.y)-1 : y-2;
    int yPrev = ((y-1) < 0)? (gridDim.y * blockDim.y)-1 : y-1;
    int yNext = ((y+1) >= (gridDim.y * blockDim.y))? 0 : y+1;
    int yNext2 = ((y+2) >= (gridDim.y * blockDim.y))? 0 : y+2;

	// compute thread indices
    unsigned int idx[25];
    idx[0] = yPrev2 * width + xPrev2;
    idx[1] = yPrev2 * width + xPrev;
    idx[2] = yPrev2 * width + x;
    idx[3] = yPrev2 * width + xNext;
    idx[4] = yPrev2 * width + xNext2;
    idx[5] = yPrev * width + xPrev2;
    idx[6] = yPrev * width + xPrev;
    idx[7] = yPrev * width + x;
    idx[8] = yPrev * width + xNext;
    idx[9] = yPrev * width + xNext2;
    idx[10] = y* width + xPrev2;
    idx[11] = y* width + xPrev;
    idx[12] = y* width + x;
    idx[13] = y* width + xNext;
    idx[14] = y* width + xNext2;
    idx[15] = yNext* width + xPrev2;
    idx[16] = yNext* width + xPrev;
    idx[17] = yNext* width + x;
    idx[18] = yNext* width + xNext;
    idx[19] = yNext* width + xNext2;
    idx[20] = yNext2* width + xPrev2;
    idx[21] = yNext2* width + xPrev;
    idx[22] = yNext2* width + x;
    idx[23] = yNext2* width + xNext;
    idx[24] = yNext2* width + xNext2;

	// prepare weights
    float weights[25];
    weights[0] = 2;
    weights[1] = 7;
    weights[2] = 12;
    weights[3] = 7;
    weights[4] = 2;

    weights[5] = 7;
    weights[6] = 31;
    weights[7] = 52;
    weights[8] = 31;
    weights[9] = 7;

    weights[10] = 15;
    weights[11] = 52;
    weights[12] = 127;
    weights[13] = 52;
    weights[14] = 15;

    weights[15] = 7;
    weights[16] = 31;
    weights[17] = 52;
    weights[18] = 31;
    weights[19] = 7;

    weights[20] = 2;
    weights[21] = 7;
    weights[22] = 12;
    weights[23] = 7;
    weights[24] = 2;

	// perform gauss kernel
    float4 src = make_float4(0,0,0,0);
    for( unsigned int p=0; p<25; ++p )
    {
        uchar4 texValue = tex1Dfetch( gaussTex, idx[p] );
    
        src.x += weights[p] * texValue.x;
        src.y += weights[p] * texValue.y;
        src.z += weights[p] * texValue.z;
    }

    src.x /= 423.0f;
    src.y /= 423.0f;
    src.z /= 423.0f;

    src.x = clamp( src.x, 0, 255.0f );
    src.y = clamp( src.y, 0, 255.0f );
    src.z = clamp( src.z, 0, 255.0f );

	// write result
    trg[idx[12]] = 
		make_uchar4( 
			(unsigned char)(src.x), 
			(unsigned char)(src.y),
			(unsigned char)(src.z),
			255);
}

 
//-------------------------------------------------------------------------
__global__ 
void swapKernel( uchar4* trg ) 
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
    float4 src = tex2D( swapTex, texCoord.x, texCoord.y );
    // swap channels
    trg[trgIdx] = make_uchar4( 
                        (unsigned char)(src.z*255.0f), 
                        (unsigned char)(src.x*255.0f),
                        (unsigned char)(src.y*255.0f),
                        (unsigned char)(src.w*255.0f));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
extern "C"
void gauss( const dim3& blocks, const dim3& threads, void* trg, void* src, unsigned int byteSize )
{
    cudaError res = cudaBindTexture( 0, gaussTex, src, byteSize ); 

    // call kernel
    gaussKernel<<< blocks, threads >>>( reinterpret_cast<uchar4*>(trg) );
}

//-------------------------------------------------------------------------
extern "C"
void swap( const dim3& blocks, const dim3& threads, void* trgBuffer, void* srcArray )
{
    // set texture parameters
    swapTex.normalized = true;                      // normalized texture coordinates (element of [0:1])
    swapTex.filterMode = cudaFilterModeLinear;      // bilinear interpolation 
    swapTex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    swapTex.addressMode[1] = cudaAddressModeClamp;

    // bind texture
    cudaError res = cudaBindTextureToArray( swapTex, reinterpret_cast<cudaArray*>(srcArray) );

    // call kernel
    swapKernel<<< blocks, threads >>>( reinterpret_cast<uchar4*>(trgBuffer) );
}

#endif // TEXDEMO_TEXSTREAMER_KERNEL_H