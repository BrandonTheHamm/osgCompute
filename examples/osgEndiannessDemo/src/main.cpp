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
#include <osg/Notify>
#include <osgCuda/Buffer>
#include <osgCompute/Computation>
#include <cuda_runtime.h>

extern "C" void swapEndianness( unsigned int numBytes, void* bytes );

class SwapComputation : public osgCompute::Computation
{
public:
    virtual void launch()
    {
        if( !_buffer.valid() )
            return;

        swapEndianness( _buffer->getNumElements(), _buffer->map( osgCompute::MAP_DEVICE_TARGET ) );
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        _buffer = dynamic_cast<osgCompute::Memory*>( &resource );
    }

protected:
    osg::ref_ptr<osgCompute::Memory> _buffer;
};

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::INFO );

    // You can use modules and buffers in the update cycle or everywhere
    // you want even with no OpenGL interoperability
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
    osg::ref_ptr<osgCuda::Buffer> buffer = new osgCuda::Buffer;
    buffer->setElementSize( sizeof(unsigned int) );
    buffer->setDimension(0, numEndians);

    ///////////////////
    // LAUNCH MODULE //
    ///////////////////
    osg::ref_ptr<SwapComputation> module = new SwapComputation;
    module->acceptResource( *buffer );

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
