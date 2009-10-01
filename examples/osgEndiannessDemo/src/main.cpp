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
#include <osgCuda/Buffer>
#include <osgCuda/Context>
#include <osgCompute/Module>

extern "C"
void swapEndianness( unsigned int numBlocks, unsigned int numThreads, void* bytes );

class SwapModule : public osgCompute::Module
{
public:
    SwapModule() : osgCompute::Module() {clearLocal();}

    META_Module( , SwapModule )
    virtual bool init();
    virtual void clear() { clearLocal(); osgCompute::Module::clear(); }
    inline void setBuffer( osgCompute::Buffer* buffer ) { _buffer = buffer; }
    inline osgCompute::Buffer* getBuffer() { return _buffer; }

    virtual void launch( const osgCompute::Context& context ) const;

protected:
    virtual ~SwapModule() { clearLocal(); }
    void clearLocal() { _buffer = NULL; }

    unsigned int                                     _numThreads;
    unsigned int                                     _numBlocks;
    osgCompute::Buffer*                              _buffer;

private:
    SwapModule(const SwapModule&, const osg::CopyOp& ) {}
    inline SwapModule &operator=(const SwapModule &) { return *this; }
};

void SwapModule::launch( const osgCompute::Context& context ) const
{
    void* bufferPtr = _buffer->map( context );
    swapEndianness( _numBlocks, _numThreads, bufferPtr );
}

bool SwapModule::init()
{
    if( !_buffer )
        return false;

    _numThreads = 1;
    _numBlocks = _buffer->getDimension(0) / _numThreads;

    return osgCompute::Module::init();
}


int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::INFO );

    unsigned int bigEndians[] = { 0x3faff7b4, 0x32332323, 0xffccaadd, 0xaaaacccc };
    unsigned int numEndians = sizeof(bigEndians)/sizeof(unsigned int);

    // create context
    osg::ref_ptr<osgCompute::Context> context = new osgCuda::Context;
    if( !context.valid() )
        return -1;
    context->setId( 0 );

    // activate context
    context->apply();

    // create buffer
    osg::ref_ptr<osgCuda::Buffer> buffer = new osgCuda::Buffer;
	buffer->setElementSize( sizeof(unsigned int) );
    buffer->setDimension(0, numEndians);
    buffer->init();


    // create module
    osg::ref_ptr<SwapModule> module = new SwapModule;
    if( !module.valid() )
        return -1;

    module->setBuffer( buffer.get() );
    module->init();

    // print numbers
    osg::notify(osg::INFO)<<"Before conversion: "<<std::endl;
    for( unsigned int v=0; v<numEndians; ++v )
        osg::notify(osg::INFO)<<std::hex<< bigEndians[v] <<std::endl;


    unsigned int* bufferPtr = (unsigned int*)buffer->map( *context, osgCompute::MAP_HOST_TARGET );
    memcpy( bufferPtr, bigEndians, sizeof(bigEndians) );


    ///////////////////
    // LAUNCH MODULE //
    ///////////////////
    module->launch( *context );


    // print result
    bufferPtr = (unsigned int*)buffer->map( *context, osgCompute::MAP_HOST_SOURCE );
    osg::notify(osg::INFO)<<std::endl<<"After conversion: "<<std::endl;
    for( unsigned int v=0; v<buffer->getDimension(0); ++v )
        osg::notify(osg::INFO)<<std::hex<< bufferPtr[v] <<std::endl;

    return 0;
}
