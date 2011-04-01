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
#include <memory.h>
#include <osg/Notify>
#include <osgCuda/Memory>
#include <osgCompute/Module>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
extern "C" void swapEndianness( unsigned int numBlocks, unsigned int numThreads, void* bytes );

/**
*/
class SwapModule : public osgCompute::Module
{
public:
    SwapModule() : osgCompute::Module() {clearLocal();}

    META_Object( , SwapModule )
        virtual bool init();
    virtual void launch();

    inline void setBuffer( osgCompute::Memory* buffer ) { _buffer = buffer; }

    virtual void clear() { clearLocal(); osgCompute::Module::clear(); }
protected:
    virtual ~SwapModule() { clearLocal(); }
    void clearLocal() { _buffer = NULL; }

    unsigned int                                     _numThreads;
    unsigned int                                     _numBlocks;
    osg::ref_ptr<osgCompute::Memory>                 _buffer;

private:
    SwapModule(const SwapModule&, const osg::CopyOp& ) {}
    inline SwapModule &operator=(const SwapModule &) { return *this; }
};

//------------------------------------------------------------------------------
void SwapModule::launch()
{
    swapEndianness( _numBlocks, _numThreads, _buffer->map() );
}

//------------------------------------------------------------------------------
bool SwapModule::init()
{
    if( !_buffer )
        return false;

    _numThreads = 1;
    _numBlocks = _buffer->getDimension(0) / _numThreads;

    return osgCompute::Module::init();
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::INFO );

    // You can use modules and buffers in the update cycle or everywhere
    // you want. But please make sure that the context is still active at
    // computation time if you use osgCuda::Geometry or osgCuda::Texture objects!!!
	cudaSetDevice(0);

    ///////////////////
    // BEFORE LAUNCH //
    ///////////////////
    unsigned int bigEndians[] = { 0x3faff7b4, 0x32332323, 0xffccaadd, 0xaaaacccc };
    unsigned int numEndians = sizeof(bigEndians)/sizeof(unsigned int);

    osg::notify(osg::INFO)<<"Before conversion: "<<std::endl;
    for( unsigned int v=0; v<numEndians; ++v )
        osg::notify(osg::INFO)<<std::hex<< bigEndians[v] <<std::endl;



    // create a buffer
    osg::ref_ptr<osgCuda::Memory> buffer = new osgCuda::Memory;
    buffer->setElementSize( sizeof(unsigned int) );
    buffer->setDimension(0, numEndians);
    buffer->init();


    ///////////////////
    // LAUNCH MODULE //
    ///////////////////
    osg::ref_ptr<SwapModule> module = new SwapModule;
    if( !module.valid() )
        return -1;

    module->setBuffer( buffer.get() );
    module->init();

    // Instead of attaching an osg::Array you can map the buffer to the
    // CPU memory and fill it directly. The TARGET specifier in MAP_HOST_TARGET
    // tells osgCompute that the buffer is updated on the CPU. This has an effect
    // on later mappings of the GPU memory (e.g. MAP_DEVICE): before a pointer
    // is returned the CPU data is copied to the GPU memory.
    unsigned int* bufferPtr = (unsigned int*)buffer->map( osgCompute::MAP_HOST_TARGET );
    memcpy( bufferPtr, bigEndians, sizeof(bigEndians) );

    module->launch();

    //////////////////
    // AFTER LAUNCH //
    //////////////////
    bufferPtr = (unsigned int*)buffer->map( osgCompute::MAP_HOST_SOURCE );
    osg::notify(osg::INFO)<<std::endl<<"After conversion: "<<std::endl;
    for( unsigned int v=0; v<buffer->getDimension(0); ++v )
        osg::notify(osg::INFO)<<std::hex<< bufferPtr[v] <<std::endl;

    return 0;
}
