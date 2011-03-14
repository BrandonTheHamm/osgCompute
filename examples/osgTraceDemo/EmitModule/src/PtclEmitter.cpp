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

#include <vector_types.h>
#include <math.h>
#include <cstdlib>
#include <osg/Notify>
#include <osg/Array>
#include <osg/ref_ptr>
#include <osgCompute/Module>
#include <osgCompute/Memory>
#include <osgCuda/Buffer>

//------------------------------------------------------------------------------
extern "C"
void reseed( unsigned int numBlocks,
            unsigned int numThreads,
            void* ptcls,
            void* seeds,
            unsigned int seedCount,
            unsigned int seedIdx,
            float3 bbmin,
            float3 bbmax );


namespace PtclDemo
{
    /**
    */
    class PtclEmitter : public osgCompute::Module 
    {
    public:
        PtclEmitter() : osgCompute::Module() {clearLocal();}

        META_Object( PtclDemo, PtclEmitter )

        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

        virtual void clear() { clearLocal(); osgCompute::Module::clear(); }
    protected:
        virtual ~PtclEmitter();
        void clearLocal();

        unsigned int                                      _numBlocks;
        unsigned int                                      _numThreads;

        osg::Vec3f                                        _seedBoxMin;
        osg::Vec3f                                        _seedBoxMax;
        osg::ref_ptr<osgCompute::Memory>                  _ptcls;
        osg::ref_ptr<osgCompute::Memory>                  _seeds;

    private:
        PtclEmitter(const PtclEmitter&, const osg::CopyOp& ) {} 
        inline PtclEmitter &operator=(const PtclEmitter &) { return *this; }
    };


    //------------------------------------------------------------------------------
    PtclEmitter::~PtclEmitter()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool PtclEmitter::init()
    {
        if( !_ptcls.valid() )
        {
            osg::notify( osg::WARN )
                << "ParticleDemo::ParticleMover::init(): buffers are missing."
                << std::endl;

            return false;
        }

        _seedBoxMin = osg::Vec3f(-1.f,-1.f,-1.f);
        _seedBoxMax = osg::Vec3f(1.f,1.f,1.f);

        ////////////////////////
        // CREATE SEED BUFFER //
        ////////////////////////
        osg::FloatArray* seedValues = new osg::FloatArray();
        for( unsigned int s=0; s<_ptcls->getNumElements(); ++s )
            seedValues->push_back( float(rand()) / RAND_MAX );

        osg::ref_ptr<osgCuda::Buffer> seedBuffer = new osgCuda::Buffer;
        seedBuffer->setElementSize( sizeof(float) );
        seedBuffer->setDimension(0,_ptcls->getNumElements());
        seedBuffer->setArray( seedValues );
        _seeds = seedBuffer;

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // One Thread handles a single particle
        // buffer size must be a multiple of 128 x sizeof(float4)
        _numBlocks = _ptcls->getDimension(0) / 128;
        _numThreads = 128;

        return osgCompute::Module::init();
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::launch()
    {
        if( isClear() )
            return;

        ////////////
        // PARAMS //
        ////////////
        unsigned int seedIdx = static_cast<unsigned int>(rand());
        float3 bbmin;
        bbmin.x = _seedBoxMin.x();
        bbmin.y = _seedBoxMin.y();
        bbmin.z = _seedBoxMin.z();

        float3 bbmax;
        bbmax.x = _seedBoxMax.x();
        bbmax.y = _seedBoxMax.y();
        bbmax.z = _seedBoxMax.z();

        ////////////
        // RESEED //
        ////////////
        reseed(
            _numBlocks,
            _numThreads,
            _ptcls->map(),
            _seeds->map(),
            _seeds->getDimension(0),
            seedIdx,
            bbmin,
            bbmax );
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PtclEmitter::clearLocal()
    {
        _numBlocks = 1;
        _numThreads = 1;
        _ptcls = NULL;
        _seeds = NULL;
        _seedBoxMin = osg::Vec3f(0,0,0);
        _seedBoxMax = osg::Vec3f(0,0,0);
    }
}

//-----------------------------------------------------------------------------
// Use this function to return a new warp module to the application
extern "C" OSGCOMPUTE_MODULE_EXPORT osgCompute::Module* OSGCOMPUTE_CREATE_MODULE_FUNCTION( void ) 
{
    return new PtclDemo::PtclEmitter;
}
