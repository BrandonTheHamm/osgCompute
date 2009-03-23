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
#include <osgCuda/Module>

extern "C"
void swapEndianness( unsigned int numBlocks, unsigned int numThreads, unsigned int* bytes );

class SwapModule : public osgCuda::Module
{
public:
    SwapModule() : osgCuda::Module() {clearLocal();}

    META_Module( , SwapModule )
    virtual bool init();
    virtual void clear() { clearLocal(); osgCuda::Module::clear(); }
    inline void setBuffer( osgCuda::UIntBuffer* buffer ) { _buffer = buffer; }
    inline osgCuda::UIntBuffer* getBuffer() { return _buffer.get(); }

    virtual void launch( const osgCompute::Context& context ) const;

protected:
    virtual ~SwapModule() { clearLocal(); }
    void clearLocal() { _buffer = NULL; }

    unsigned int                                     _numThreads;
    unsigned int                                     _numBlocks;
    osg::ref_ptr<osgCuda::UIntBuffer>                _buffer;

private:
    SwapModule(const SwapModule&, const osg::CopyOp& ) {}
    inline SwapModule &operator=(const SwapModule &) { return *this; }
};

void SwapModule::launch( const osgCompute::Context& context ) const
{
    const osgCuda::Context* cudaContext = dynamic_cast<const osgCuda::Context*>( &context );
    if( !cudaContext ||
        !cudaContext->getDeviceProperties() ||
        _numBlocks > static_cast<unsigned int>( cudaContext->getDeviceProperties()->maxGridSize[1] ) )
        return;

    unsigned int* bufferPtr = _buffer->map( context, osgCompute::MAP_DEVICE );
    swapEndianness( _numBlocks, _numThreads, bufferPtr );
    _buffer->unmap( context );
}

bool SwapModule::init()
{
    if( !_buffer.valid() )
        return false;

    _numThreads = 1;
    _numBlocks = _buffer->getDimension(0) / _numThreads;

    return osgCuda::Module::init();
}


int main(int argc, char *argv[])
{
    osg::setNotifyLevel( osg::INFO );

    unsigned int bigEndians[] = { 0x3faff7b4, 0x32332323, 0xffccaadd, 0xaaaacccc };
    unsigned int numEndians = sizeof(bigEndians)/sizeof(unsigned int);

    // create context
    osg::ref_ptr<osgCompute::Context> context = osgCompute::Context::createInstance( 0, "osgCuda", "Context" );
    if( !context.valid() )
        return -1;

    // activate context
    context->apply();

    // create buffer
    osg::ref_ptr<osgCuda::UIntBuffer> buffer = new osgCuda::UIntBuffer;
    buffer->setDimension(0, numEndians);
    buffer->init();


    // create module
    osg::ref_ptr<SwapModule> module = new SwapModule;
    if( !module.valid() )
        return -1;

    module->setBuffer( buffer );
    module->init();



    // print numbers
    osg::notify(osg::INFO)<<"Before conversion: "<<std::endl;
    for( unsigned int v=0; v<numEndians; ++v )
        osg::notify(osg::INFO)<<std::hex<< bigEndians[v] <<std::endl;


    unsigned int* bufferPtr = buffer->map( *context, osgCompute::MAP_HOST_TARGET );
    memcpy( bufferPtr, bigEndians, sizeof(bigEndians) );
    buffer->unmap( *context );


    ///////////////////
    // LAUNCH MODULE //
    ///////////////////
    module->launch( *context );


    // print result
    bufferPtr = buffer->map( *context, osgCompute::MAP_HOST_SOURCE );
    osg::notify(osg::INFO)<<std::endl<<"After conversion: "<<std::endl;
    for( unsigned int v=0; v<buffer->getDimension(0); ++v )
        osg::notify(osg::INFO)<<std::hex<< bufferPtr[v] <<std::endl;

    return 0;
}
