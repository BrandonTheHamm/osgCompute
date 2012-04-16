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

texture<uchar4, 1, cudaReadModeElementType> srcTex;

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
void kSobelFilter( uchar4* trg ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    int xPrev = ((x-1) < 0)? ((gridDim.x) * blockDim.x)-1 : x-1;
    int xNext = ((x+1) >= (gridDim.x * blockDim.x))? 0 : x+1;
    int yPrev = ((y-1) < 0)? (gridDim.y * blockDim.y)-1 : y-1;
    int yNext = ((y+1) >= (gridDim.y * blockDim.y))? 0 : y+1;

    unsigned int idx[9];
    idx[0] = yPrev * width + xPrev;
    idx[1] = yPrev * width + x;
    idx[2] = yPrev * width + xNext;
    idx[3] = y * width + xPrev;
    idx[4] = y * width + x;
    idx[5] = y * width + xNext;
    idx[6] = yNext * width + xPrev;
    idx[7] = yNext * width + x;
    idx[8] = yNext * width + xNext;

    float weightsX[9];
    weightsX[0] = 1;
    weightsX[1] = 0;
    weightsX[2] = -1;
    weightsX[3] = 2;
    weightsX[4] = 0;
    weightsX[5] = -2;
    weightsX[6] = 1;
    weightsX[7] = 0;
    weightsX[8] = -1;

    float weightsY[9];
    weightsY[0] = 1;
    weightsY[1] = 2;
    weightsY[2] = 1;
    weightsY[3] = 0;
    weightsY[4] = 0;
    weightsY[5] = 0;
    weightsY[6] = -1;
    weightsY[7] = -2;
    weightsY[8] = -1;

    float4 srcX = make_float4(0,0,0,0);
    for( unsigned int p=0; p<9; ++p )
    {
        uchar4 texValue = tex1Dfetch( srcTex, idx[p] );
    
        srcX.x += weightsX[p] * texValue.x;
        srcX.y += weightsX[p] * texValue.y;
        srcX.z += weightsX[p] * texValue.z;
    }

    float4 srcY = make_float4(0,0,0,0);
    for( unsigned int p=0; p<9; ++p )
    {
        uchar4 texValue = tex1Dfetch( srcTex, idx[p] );
    
        srcY.x += weightsY[p] * texValue.x;
        srcY.y += weightsY[p] * texValue.y;
        srcY.z += weightsY[p] * texValue.z;
    }

    float4 src;
    src.x = clamp( sqrt(srcX.x * srcX.x + srcY.x * srcY.x), 0, 255 );
    src.y = clamp( sqrt(srcX.y * srcX.y + srcY.y * srcY.y), 0, 255 );
    src.z = clamp( sqrt(srcX.z * srcX.z + srcY.z * srcY.z), 0, 255 );

   
    trg[idx[4]] = 
    make_uchar4( 
    (unsigned char)(src.x), 
    (unsigned char)(src.y),
    (unsigned char)(src.z),
    255);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
extern "C"
void sobelFilter( unsigned int numPixelsX, unsigned int numPixelsY, void* trgBuffer, void* srcBuffer, unsigned int srcBufferSize )
{
    dim3 threads = dim3( 16, 16, 1 );
    dim3 blocks = dim3( numPixelsX/16, numPixelsY/16, 1 );

    cudaBindTexture( 0, srcTex, srcBuffer, srcBufferSize ); 
    kSobelFilter<<< blocks, threads >>>( reinterpret_cast<uchar4*>(trgBuffer) );
}
