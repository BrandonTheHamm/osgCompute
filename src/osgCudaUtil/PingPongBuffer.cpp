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
	}

	//------------------------------------------------------------------------------
	bool PingPongBuffer::init()
	{
		if( !isClear() )
			return true;

		if( _bufferStack.empty() )
		{
			osg::notify(osg::WARN) 
				<< "PingPongBuffer::init() of buffer \""<<getName()<<"\": No buffers found."
				<< std::endl;

			clear();
			return false;
		}

		//////////////////
		// INIT BUFFERS //
		//////////////////
		for( unsigned int b=0; b<_bufferStack.size(); ++b )
		{
			if( !_bufferStack.at(b).valid() )
				continue;

			osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack.at(b).get());
			if( curBuffer->isClear() && !curBuffer->init() )
			{
				osg::notify(osg::WARN) 
					<< "PingPongBuffer::init() of buffer \""<<getName()<<"\": Buffer at idx \""
					<<b<< "\"could not be initialized."
					<< std::endl;

				clear();
				return false;
			}
		}

		//////////////////////
		// SETUP PARAMETERS //
		//////////////////////
		if( getElementSize() == 0 )
		{
			if( _bufferStack.size() < _refIdx ||
				!_bufferStack.at(_refIdx).valid() )
			{
				osg::notify(osg::WARN) 
					<< "PingPongBuffer::init() of buffer \""<<getName()<<"\": No element size set and no reference buffer has been found."
					<< std::endl;

				clear();
				return false;
			}
			else
			{
				osgCompute::Memory* refBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack.at(_refIdx).get());
				setElementSize( refBuffer->getElementSize() );
			}
		}


		if( getNumDimensions() == 0 )
		{
			if( _bufferStack.size() < _refIdx ||
				NULL == _bufferStack.at(_refIdx) )
			{
				osg::notify(osg::WARN) 
					<< "PingPongBuffer::init() of buffer \""<<getName()<<"\": No dimensions set but no reference buffer has been found."
					<< std::endl;

				clear();
				return false;
			}
			else
			{
				osgCompute::Memory* refBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack.at(_refIdx).get());
				for( unsigned int d=0; d<refBuffer->getNumDimensions(); ++d )
					setDimension( d, refBuffer->getDimension(d) );
			}
		}

		///////////////////
		// SETUP BUFFERS //
		///////////////////
		unsigned int byteSize = getElementSize();
		for( unsigned int d=0; d<getNumDimensions(); ++d )
			byteSize *= getDimension(d);

		for( unsigned int b=0; b<_bufferStack.size(); ++b )
		{
            if( !_bufferStack.at(b).valid() )
            {
                osg::notify(osg::WARN) 
					<< "PingPongBuffer::init() of buffer \""<<getName()<<"\": no buffer specified on idx "<<b<<"."
					<< std::endl;

				clear();
				return false;
            }
			// Check the byte size to be consistent 
			// among buffers
			osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack.at(b).get());
			if( byteSize != curBuffer->getByteSize( osgCompute::MAP_DEVICE ) )
			{
				osg::notify(osg::WARN) 
					<< "PingPongBuffer::init() of buffer \""<<getName()<<"\": Buffers have different sizes."
					<< std::endl;

				clear();
				return false;
			}
		}

		return osgCompute::Memory::init();
	}

    //------------------------------------------------------------------------------
    bool PingPongBuffer::createSwapBuffers()
    {
        if( getNumDimensions() == 0 && getElementSize() == 0 )
		    return false;
        
        for( unsigned int b=0; b<_bufferStack.size(); ++b )
        {
            // Create Cuda::Buffer if no buffer is specified
            // on this stack entry
            if( !_bufferStack.at(b).valid() )
            {
                osg::ref_ptr<osgCuda::Memory> newBuffer = new osgCuda::Memory;
                newBuffer->setElementSize( getElementSize() );
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
		if( isClear() )
			if( !init() )
				return NULL;

		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( !_bufferStack[mapIdx].valid() )
			return NULL;

		osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack[mapIdx].get());
		return curBuffer->map( mapping, offset );		
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::unmap( unsigned int bufferIdx /*= 0 */ )
	{
		if( isClear() )
			if( !init() )
				return;

		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( !_bufferStack[mapIdx].valid() )
			return;

		osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack[mapIdx].get());
		curBuffer->unmap();
	}

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getAllocatedByteSize( unsigned int mapping, unsigned int bufferIdx /*= 0 */ ) const
    {
        if( osgCompute::Resource::isClear() )
            return 0;

        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( !_bufferStack[mapIdx].valid() )
            return 0;

         return _bufferStack[mapIdx]->getAllocatedByteSize( mapping );
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getByteSize( unsigned int mapping, unsigned int bufferIdx /*= 0 */  ) const
    {
        if( osgCompute::Resource::isClear() )
            return 0;

         unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
         if( !_bufferStack[mapIdx].valid() )
             return 0;

         return _bufferStack[mapIdx]->getByteSize( mapping );
    }

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getAllElementsSize( unsigned int bufferIdx /*= 0 */  ) const
    {
        if( osgCompute::Resource::isClear() )
            return 0;

        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( !_bufferStack[mapIdx].valid() )
            return 0;

        return _bufferStack[mapIdx]->getAllElementsSize();
    }

	//------------------------------------------------------------------------------
	bool PingPongBuffer::reset( unsigned int bufferIdx /*= 0 */ )
	{
		if( isClear() )
			if( !init() )
				return false;

		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( !_bufferStack[mapIdx].valid() )
			return false;

		osgCompute::Memory* curBuffer = dynamic_cast<osgCompute::Memory*>(_bufferStack[mapIdx].get());
		return curBuffer->reset();	
	}

    //------------------------------------------------------------------------------
    bool PingPongBuffer::supportsMapping( unsigned int mapping, unsigned int bufferIdx ) const
    {
        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( !_bufferStack[mapIdx].valid() )
            return false;

        const osgCompute::Memory* curBuffer = dynamic_cast<const osgCompute::Memory*>(_bufferStack[mapIdx].get());
        return curBuffer->supportsMapping( mapping );
    }


	//------------------------------------------------------------------------------
	unsigned int PingPongBuffer::getMapping( unsigned int bufferIdx /*= 0 */ ) const
	{
		unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
		if( !_bufferStack[mapIdx].valid() )
			return osgCompute::UNMAP;

		const osgCompute::Memory* curBuffer = dynamic_cast<const osgCompute::Memory*>(_bufferStack[mapIdx].get());
		return curBuffer->getMapping();
	}

    //------------------------------------------------------------------------------
    unsigned int PingPongBuffer::getPitch( unsigned int bufferIdx /*= 0 */ ) const
    {
        unsigned int mapIdx = (_stackIdx + bufferIdx) % _bufferStack.size();
        if( !_bufferStack[mapIdx].valid() )
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
        if( !isClear() )
            return;

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
	void PingPongBuffer::setRefBuffer( unsigned int idx )
	{
		if( !isClear() )
			return;

		_refIdx = idx;
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
		if( !isClear() )
			return;

        if( _bufferStack.empty() && getElementSize() == 0 && getNumElements() == 0) 
        {
            setElementSize( buffer.getElementSize() );
            for( unsigned int d=0; d<buffer.getNumDimensions(); ++d )
            {
                setDimension(d,buffer.getDimension(d));
            }
        }

		_bufferStack.push_back( &buffer );
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::insertBuffer( unsigned int idx, osgCompute::Memory& buffer )
	{
		if( !isClear() )
			return;

        if( _bufferStack.empty() && getElementSize() == 0 && getNumDimensions() == 0) 
        {
            setElementSize( buffer.getElementSize() );
            for( unsigned int d=0; d<buffer.getNumDimensions(); ++d )
            {
                setDimension(d,buffer.getDimension(d));
            }
        }

		if( _bufferStack.size() < (idx+1) )
			_bufferStack.resize( (idx+1) );

        

		_bufferStack[idx] = &buffer;
	}

	//------------------------------------------------------------------------------
	void PingPongBuffer::removeBuffer( unsigned int idx )
	{
		if( !isClear() )
			return;

		BufferStackItr itr = _bufferStack.begin();
		for( unsigned int c=0; c<idx; ++c, ++itr );
			
		_bufferStack.erase( itr );
	}

	//------------------------------------------------------------------------------
	osgCompute::Memory* PingPongBuffer::getBufferAt( unsigned int idx )
	{
		if( idx >= _bufferStack.size() )
			return NULL;

		return _bufferStack[idx].get();
	}

	//------------------------------------------------------------------------------
	const osgCompute::Memory* PingPongBuffer::getBufferAt( unsigned int idx ) const
	{
		if( idx >= _bufferStack.size() )
			return NULL;

		return _bufferStack[idx].get();
	}

    //------------------------------------------------------------------------------
    osgCompute::Memory* PingPongBuffer::refAt( unsigned int idx )
    {
        if( idx >= _bufferStack.size() )
            return NULL;

        return _bufferStack[ idx ].get();
    }

    //------------------------------------------------------------------------------
    const osgCompute::Memory* PingPongBuffer::refAt( unsigned int idx ) const
    {
        if( idx >= _bufferStack.size() )
            return NULL;

        return _bufferStack[ idx ].get();
    }


	//------------------------------------------------------------------------------
	void PingPongBuffer::clear()
	{
		osgCompute::Memory::clear();
		clearLocal();
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



}