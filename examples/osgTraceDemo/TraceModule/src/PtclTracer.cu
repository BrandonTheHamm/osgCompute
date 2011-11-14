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

#include <math_constants.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define GAM 0.003f
#define VEL_STRENGTH  10.0f 

//------------------------------------------------------------------------------
inline __device__ 
float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

//------------------------------------------------------------------------------
inline __device__ 
float4 operator*(float a, float4 b)
{
    return make_float4(a * b.x, a * b.y, a * b.z,  a * b.w);
}

//------------------------------------------------------------------------------
inline __device__
unsigned int thIdx()
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int width = gridDim.x * blockDim.x;

    return y*width + x;
}

//------------------------------------------------------------------------------
// A simple vortex field of strength GAM around straight line (0,0,z)
__device__ 
inline float4 vortexField( float4 pos )
{
    float4 vel;

    float sqrad = pos.x*pos.x + pos.y*pos.y;
    vel.x = (-CUDART_PI_F * GAM * pos.y)/( sqrad );
    vel.y = (CUDART_PI_F * GAM * pos.x)/( sqrad );
    vel.z = 0.0f;
    vel.w = 0.0f;

    return VEL_STRENGTH * vel;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
__global__
void traceKernel( float4* ptcls, float etime, unsigned int numPtcls )
{
    unsigned int ptclIdx = thIdx();
    if( ptclIdx < numPtcls )
    {
        float4 ptclPos = ptcls[ptclIdx];

        // 4th order Runge-Kutta 
        float halfETime = etime * 0.5f;
        float4 k0 = vortexField( ptclPos );
        float4 k1 = vortexField( ptclPos + (halfETime * k0) );
        float4 k2 = vortexField( ptclPos + (halfETime * k1) );
        float4 k3 = vortexField( ptclPos + (etime * k2) );

        // Advance
        ptclPos = ptclPos + etime*(1.0f/6.0f)* ( k0 + (2.0f*k1) + (2.0f*k2) + k3 );
        ptclPos.w = 1.0f;
        
        // Forward-Euler
        // ptclPos = ptclPos + etime* vortexField(ptclPos);
        // ptclPos.w = 1.0f;
        ptcls[ptclIdx] = ptclPos;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
extern "C" __host__
void trace( unsigned int numBlocks, unsigned int numThreads, void* ptcls, float etime, unsigned int numPtcls )
{
    dim3 blocks( numBlocks, 1, 1 );
    dim3 threads( numThreads, 1, 1 );

    traceKernel<<< blocks, threads >>>( (float4*) ptcls,etime, numPtcls);
}
