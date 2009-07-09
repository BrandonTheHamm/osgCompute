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
#include <osgCompute/Buffer>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    BufferStream::BufferStream() 
        :   _mapping( UNMAPPED ),
            _allocHint(0)
    {
    }

    //------------------------------------------------------------------------------
    BufferStream::~BufferStream() 
    {
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Buffer::Buffer() 
        : Resource()
    { 
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Buffer::clear()
    {
        Resource::clear();
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool Buffer::init()
    {
        if( !isClear() )
            return true;

        if( _dimensions.empty() )
        {
            osg::notify(osg::FATAL)  
                << "Buffer::init() for Buffer \""<<asObject()->getName()<<"\": No Dimensions specified."                  
                << std::endl;

            return false;
        }

        ///////////////////////
        // COMPUTE BYTE SIZE //
        ///////////////////////
        _numElements = 1;
        for( unsigned int d=0; d<_dimensions.size(); ++d )
            _numElements *= _dimensions[d];

        return Resource::init();
    }

    //------------------------------------------------------------------------------
    unsigned int Buffer::getMapping( const osgCompute::Context& context ) const
    {
        if( isClear() )
            return osgCompute::UNMAPPED;

        BufferStream* stream = lookupStream( context );
        if( NULL == stream )
        {
            osg::notify(osg::FATAL)  
                << "Buffer::getMapping() for Buffer \""
                << asObject()->getName() <<"\": Could not receive BufferStream for Context \""
                << context.getId() << "\"."
                << std::endl;

            return osgCompute::UNMAPPED;
        }

        return stream->_mapping;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    bool Buffer::init( const Context& context ) const
    {
        if( _streams.size()<=context.getId() )
            _streams.resize(context.getId()+1,NULL);

        // Allocate stream array for context
        if( NULL == _streams[context.getId()] )
        {
            _streams[context.getId()] = newStream( context );

            if( NULL == _streams[context.getId()] )
            {
                osg::notify( osg::FATAL )  
                    << "Buffer::init( \"CONTEXT\" ) for Buffer \"" << asObject()->getName()
                    << "\": DataArray could be allocated for context \"" 
                    << context.getId() << "\"."
                    << std::endl;

                return false;
            }

            _streams[context.getId()]->_context = const_cast<osgCompute::Context*>( &context );
            _streams[context.getId()]->_allocHint = getAllocHint();
        }

        // Register param if valid stream-array 
        // is allocated
        return Resource::init( context );
    }

    //------------------------------------------------------------------------------
    void Buffer::clear( const Context& context ) const
    {
        if( _streams.size() > context.getId() &&
            NULL != _streams[context.getId()] )
        {
            // delete stream
            delete _streams[context.getId()];
            _streams[context.getId()] = NULL;
        }

        // Unregister context
        return Resource::clear( context );
    }
}
