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
#include <osg/FrameStamp>
#include <osgCompute/Computation>
#include <osgCompute/Memory>

//------------------------------------------------------------------------------
extern "C"
void trace( unsigned int numBlocks, unsigned int numThreads, void* ptcls, float etime );

namespace PtclDemo
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // DECLARATION ///////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    class PtclTracer : public osgCompute::Computation 
    {
    public:
        PtclTracer() : osgCompute::Computation() {clearLocal();}

        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

        virtual void clear() { clearLocal(); osgCompute::Computation::clear(); }
    protected:
        virtual ~PtclTracer() {clearLocal();}
        void clearLocal();

        unsigned int                        _numBlocks;
        unsigned int                        _numThreads;

        osg::ref_ptr<osgCompute::Memory>    _ptcls;

    private:
        PtclTracer(const PtclTracer&, const osg::CopyOp& ) {} 
        inline PtclTracer &operator=(const PtclTracer &) { return *this; }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    bool PtclTracer::init() 
    { 
        if( !_ptcls )
        {
            osg::notify( osg::WARN ) 
                << "PtclDemo::PtclTracer::init(): particle buffer is missing."
                << std::endl;

            return false;
        }

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // One Thread handles a single particle
        // buffer size must be a multiple of 128 x sizeof(float4)
        _numBlocks = _ptcls->getDimension(0) / 128;
        _numThreads = 128;

        return osgCompute::Computation::init();
    }

    //------------------------------------------------------------------------------  
    void PtclTracer::launch()
    {
        if( isClear() )
            return;

        ////////////////////
        // MOVE PARTICLES //
        ////////////////////
        trace( _numBlocks, _numThreads, _ptcls->map(), 0.009f );
    }

    //------------------------------------------------------------------------------
    void PtclTracer::acceptResource( osgCompute::Resource& resource )
    {
        // Search for the particle buffer
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------  
    void PtclTracer::clearLocal() 
    { 
        _numBlocks = 1;
        _numThreads = 1;

        _ptcls = NULL;
    }
}

//-----------------------------------------------------------------------------
// Use this function to return a new warp module to the application
extern "C" OSGCOMPUTE_COMPUTATION_EXPORT osgCompute::Computation* OSGCOMPUTE_CREATE_COMPUTATION_FUNCTION() 
{
    return new PtclDemo::PtclTracer;
}
