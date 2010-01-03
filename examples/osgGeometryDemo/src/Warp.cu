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

#ifndef GEOMETRY_KERNEL_H
#define GEOMETRY_KERNEL_H 1


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
inline __device__ 
float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
__global__
void kWarp( float3* vertices, unsigned int numVertices, float3* initPos, float3* initNormals, float simTime )
{
    unsigned int vertIdx = thIdx();
    if( vertIdx >= numVertices )
        return; // stop if not a valid index

    float3 curVert = initPos[vertIdx];
    float3 curNorm = initNormals[vertIdx];

    // displace the vertex position
    float maxDis = 0.2f;
    float displacement = maxDis * sinf(2.0f*simTime) + maxDis;
    curVert = curVert + make_float3(displacement * curNorm.x, displacement * curNorm.y, displacement * curNorm.z);
    vertices[vertIdx] = curVert;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
extern "C" __host__
void warp(unsigned int numBlocks, 
          unsigned int numThreads, 
          void* vertices,
          unsigned int numVertices,
          void* initPos,
          void* initNormals,
          float simTime )
{
    dim3 blocks( numBlocks, 1, 1 );
    dim3 threads( numThreads, 1, 1 );

    kWarp<<< blocks, threads >>>(
        reinterpret_cast<float3*>(vertices),
        numVertices,
        reinterpret_cast<float3*>(initPos),
        reinterpret_cast<float3*>(initNormals),
        simTime );
}

#endif // GEOMETRY_KERNEL_H
