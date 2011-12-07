#include <osg/Notify>
#include <osgCuda/Memory>
#include <osgCudaUtil/PingPongBuffer>

namespace osgCuda
{   
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	PingPongBuffer::PingPongBuffer()
		: osgCompute::Memory()
	{
        clearLocal();
        // Please note that virtual functions className() and libraryName() are called
        // during observeResource() which will only develop until this class.
        // However if contructor of a subclass calls this function again observeResource
        // will change the className and libraryName of the observed pointer.
        osgCompute::ResourceObserver::instance()->observeResource( *this );
	}

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getElementSize() const
    {
        if( (_stackIdx >= _bufferStack.size()) || !_bufferStack[_stackIdx].valid() ) 
            return osgCompute::Memory::getElementSize();

        return _bufferStack[_stackIdx]->getElementSize();
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getDimension( unsigned int dimIdx ) const
    {
        if( (_stackIdx >= _bufferStack.size()) || !_bufferStack[_stackIdx].valid() ) 
            return osgCompute::Memory::getDimension( dimIdx );

        return _bufferStack[_stackIdx]->getDimension(dimIdx);
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getNumDimensions() const
    {
        if( (_stackIdx >= _bufferStack.size()) || !_bufferStack[_stackIdx].valid() ) 
            return osgCompute::Memory::getNumDimensions();

        return _bufferStack[_stackIdx]->getNumDimensions();
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getNumElements() const
    {
        if( (_stackIdx >= _bufferStack.size()) || !_bufferStack[_stackIdx].valid() ) 
            return osgCompute::Memory::getNumElements();

        return _bufferStack[_stackIdx]->getNumElements();
    }


    //------------------------------------------------------------------------------
    bool PingPongBuffer::createSwapBuffers()
    {
        if( getSwapCount() == 0 )
		    return false;
        
        for( unsigned int b=0; b<_bufferStack.size(); ++b )
        {
            // Create Cuda::Buffer if no buffer is specified
            // on this stack entry
            if( !_bufferStack.at(b).valid() )
            {
                osg::ref_ptr<osgCuda::Memory> newBuffer = new osgCuda::Memory;
                newBuffer->setElementSize( osgCompute::Memory::getElementSize() );
                for( unsigned int d=0; d<getNumDimensions(); ++d )
                    newBuffer->setDimension(d,getDimension(d));

                _bufferStack.at(b) = newBuffer;
            }
        }

        return true;
    }

	//------------------------------------------------------------------------------
	void* PingPongBuffer::map( unsigned int mapping/* = osgCompute::MAP_DEVICE*/, unsigned int offset/* = 0*/, unsigned int bufferIdx /*= 0 */ )
	{
		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( !_bufferStack[mapIdx].valid() )
			return NULL;

		osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack[mapIdx].get());
		return curBuffer->map( mapping, offset );		
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::unmap( unsigned int bufferIdx /*= 0 */ )
	{
		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( !_bufferStack[mapIdx].valid() )
			return;

		osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack[mapIdx].get());
		curBuffer->unmap();
	}

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getAllocatedByteSize( unsigned int mapping, unsigned int bufferIdx /*= 0 */ ) const
    {
        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
            return osgCompute::Memory::getAllocatedByteSize( mapping );

         return _bufferStack[mapIdx]->getAllocatedByteSize( mapping );
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getByteSize( unsigned int mapping, unsigned int bufferIdx /*= 0 */  ) const
    {
         unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
         if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
             return osgCompute::Memory::getByteSize( mapping );

         return _bufferStack[mapIdx]->getByteSize( mapping );
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getAllElementsSize( unsigned int bufferIdx /*= 0 */  ) const
    {
        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
            return osgCompute::Memory::getAllElementsSize();

        return _bufferStack[mapIdx]->getAllElementsSize();
    }

	//------------------------------------------------------------------------------
	bool PingPongBuffer::reset( unsigned int bufferIdx /*= 0 */ )
	{
		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
			return false;

		osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack[mapIdx].get());
		return curBuffer->reset();	
	}

    //------------------------------------------------------------------------------
    bool PingPongBuffer::supportsMapping( unsigned int mapping, unsigned int bufferIdx ) const
    {
        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
            return false;

        const osgCompute::Memory* curBuffer = dynamic_cast<const osgCompute::Memory*>(_bufferStack[mapIdx].get());
        return curBuffer->supportsMapping( mapping );
    }


	//------------------------------------------------------------------------------
	unsigned int PingPongBuffer::getMapping( unsigned int bufferIdx /*= 0 */ ) const
	{
		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
			return osgCompute::UNMAP;

		const osgCompute::Memory* curBuffer = dynamic_cast<const osgCompute::Memory*>(_bufferStack[mapIdx].get());
		return curBuffer->getMapping();
	}

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getPitch( unsigned int bufferIdx /*= 0 */ ) const
    {
        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( _bufferStack.empty() || !_bufferStack[mapIdx].valid() )
            return 0;

        const osgCompute::Memory* curBuffer = dynamic_cast<const osgCompute::Memory*>(_bufferStack[mapIdx].get());
        return curBuffer->getPitch();
    }

	//------------------------------------------------------------------------------
	void PingPongBuffer::swap( unsigned int incr /*= 1 */ )
	{
		_stackIdx += incr;
		_stackIdx %= _bufferStack.size();
	}

    //------------------------------------------------------------------------------
    void PingPongBuffer::setSwapCount( unsigned int sc )
    {
        if( _bufferStack.size() <= sc )
            _bufferStack.resize( sc );
    }

	//------------------------------------------------------------------------------
	unsigned int PingPongBuffer::getSwapCount() const
	{
		return _bufferStack.size();
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::setSwapIdx( unsigned int idx )
	{
		_stackIdx = idx;
		_stackIdx %= _bufferStack.size();
	}

	//------------------------------------------------------------------------------
	unsigned int PingPongBuffer::getSwapIdx() const
	{
		return _stackIdx;
	}

	//------------------------------------------------------------------------------
	BufferStack& PingPongBuffer::getBuffers()
	{
		return _bufferStack;
	}

	//------------------------------------------------------------------------------
	const BufferStack& PingPongBuffer::getBuffers() const
	{
		return _bufferStack;
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::appendBuffer( osgCompute::Memory& buffer )
	{
		_bufferStack.push_back( &buffer );
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::insertBuffer( unsigned int stackIdx, osgCompute::Memory& buffer )
    {
		if( _bufferStack.size() < (stackIdx+1) )
			_bufferStack.resize( (stackIdx+1) );

		_bufferStack[stackIdx] = &buffer;
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::removeBuffer( unsigned int stackIdx )
    {
		BufferStackItr itr = _bufferStack.begin();
		for( unsigned int c=0; c<stackIdx; ++c, ++itr );
			
		_bufferStack.erase( itr );
	}

    //------------------------------------------------------------------------------
    osgCompute::Memory* PingPongBuffer::getBufferAt( unsigned int stackIdx )
    {
        if( stackIdx >= _bufferStack.size() )
            return NULL;

        return _bufferStack[stackIdx].get();
    }

    //------------------------------------------------------------------------------
    const osgCompute::Memory* PingPongBuffer::getBufferAt( unsigned int stackIdx ) const
    {
        if( stackIdx >= _bufferStack.size() )
            return NULL;

        return _bufferStack[stackIdx].get();
    }

    //------------------------------------------------------------------------------
    void PingPongBuffer::releaseObjects()
    {
        // Do not call 
        // _bufferStack[s]->releaseObjects();
        for( unsigned int s=0; s<_bufferStack.size(); ++s )
            _bufferStack[s]->releaseObjects();
    }

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//------------------------------------------------------------------------------
	PingPongBuffer::~PingPongBuffer()
	{
		clearLocal();
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::clearLocal()
	{
		_bufferStack.clear();
		_stackIdx = 0;
		_refIdx = 0;
	} 

	//------------------------------------------------------------------------------
    unsigned int PingPongBuffer::computePitch() const
    {
        // should not be called
        return 0;
    }

    //------------------------------------------------------------------------------
    void PingPongBuffer::clear()
    {
        for( unsigned int s=0; s<_bufferStack.size(); ++s )
            _bufferStack[s]->clear();
    }
}