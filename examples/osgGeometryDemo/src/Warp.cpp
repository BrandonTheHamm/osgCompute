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
#include <osg/Array>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>
#include "Warp"

// Declare CUDA-kernel functions
extern "C"
void warp(unsigned int numBlocks, 
          unsigned int numThreads, 
          void* vertices,
          unsigned int numVertices,
          void* initPos,
          void* initNormals,
          float simTime );

namespace GeometryDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    bool Warp::init() 
    { 
        if( !_vertices.valid() )
        {
            osg::notify( osg::WARN )
                << "GeometryDemo::Warp::init(): buffer is missing."
                << std::endl;

            return false;
        }

        osg::FrameStamp* fs = (osg::FrameStamp*) getUserData();
        if( !fs )
        {
            osg::notify( osg::WARN )
                << "GeometryDemo::Warp::init(): frame stamp is missing."
                << std::endl;

            return false;
        }

        // We need to read the geometry information.
        osgCuda::Geometry* geometry = dynamic_cast<osgCuda::Geometry*>( ((osgCompute::InteropMemory*)_vertices.get())->getInteropObject() );
        if( !geometry )
            return false;

        // Create the static reference buffers
        osg::ref_ptr<osgCuda::Buffer> normals= new osgCuda::Buffer;
        normals->setArray( geometry->getNormalArray() );
        normals->setElementSize( geometry->getNormalArray()->getDataSize() * sizeof(float) );
        normals->setDimension( 0, geometry->getNormalArray()->getNumElements() );
        if( !normals->init() )
        {
            osg::notify( osg::WARN )
                << "GeometryDemo::Warp::init(): cannot create normal array."
                << std::endl;

            return false;
        }
        _initNormals = normals;

        osg::ref_ptr<osgCuda::Buffer> positions = new osgCuda::Buffer;
        positions->setArray( geometry->getVertexArray() );
        positions->setElementSize( geometry->getVertexArray()->getDataSize() * sizeof(float) );
        positions->setDimension( 0, geometry->getVertexArray()->getNumElements() );
        if( !positions->init() )
        {
            osg::notify( osg::WARN )
                << "GeometryDemo::Warp::init(): cannot create position array."
                << std::endl;

            return false;
        }
        _initPos = positions;

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        _numBlocks = _vertices->getDimension(0) / 128;
        if( _vertices->getDimension(0) % 128 != 0 )
            _numBlocks+=1;

        _numThreads = 128;
        return osgCompute::Module::init();
    }

    //------------------------------------------------------------------------------  
    void Warp::launch()
    {
        if( isClear() )
            return;

        float time = (float)((osg::FrameStamp*) getUserData())->getSimulationTime();

        ///////////////////
        // MOVE VERTICES //
        ///////////////////
        warp(_numBlocks,
            _numThreads,
            _vertices->map(),			
            _vertices->getNumElements(),
            _initPos->map(),
            _initNormals->map(),
            time );

        // Read out new vertex position on the CPU or change the values with
        // osgCompute::MAP_HOST_TARGET
        //_verts = (float*)_vertices->map( osgCompute::MAP_HOST_SOURCE );
    }

    //------------------------------------------------------------------------------
    void Warp::acceptResource( osgCompute::Resource& resource )
    {
        // Search for your handles. This Method is called for each resource
        // located in the subgraph of this module.
        if( resource.isIdentifiedBy( "WARP_GEOMETRY" ) )
            _vertices = dynamic_cast<osgCompute::Memory*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void Warp::clearLocal() 
    { 
        _numBlocks = 0;
        _numThreads = 0;
        _vertices = NULL;
        _initNormals = NULL;
        _initPos = NULL;
    }
}
