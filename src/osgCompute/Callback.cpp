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
#include <osg/RenderInfo>
#include <osgCompute/Callback>
#include <osgCompute/Memory>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    GLMemoryTargetCallback::GLMemoryTargetCallback()
    {

    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::observe( GLMemory* memory )
    {
        if( memory == NULL )
            return;

        if( !observesGLMemory(memory) )
        {
            _memories.push_back( memory );
        }
    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::remove( GLMemory* memory )
    {
        if( memory == NULL )
            return;

        for( std::vector< osg::observer_ptr<GLMemory> >::iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if( (*itr).get() == memory )
            {
                _memories.erase( itr );
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::remove( const std::string& identifier )
    {
        for( std::vector< osg::observer_ptr<GLMemory> >::iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if( (*itr).valid() && (*itr)->isIdentifiedBy(identifier) )
            {
                _memories.erase( itr );
                return;
            }
        }
    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::clear()
    {
        _memories.clear();
    }

    //------------------------------------------------------------------------------
    unsigned int GLMemoryTargetCallback::getNumObserved() const
    {
        return _memories.size();
    }

    //------------------------------------------------------------------------------
    bool GLMemoryTargetCallback::observesGLMemory( GLMemory* memory ) const
    {
        if( memory == NULL )
            return false;

        for( std::vector< osg::observer_ptr<GLMemory> >::const_iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if( (*itr).valid() && (*itr).get() == memory )
                return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool GLMemoryTargetCallback::observesGLMemory( const std::string& identifier ) const
    {
        for( std::vector< osg::observer_ptr<GLMemory> >::const_iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if( (*itr).valid() && (*itr)->isIdentifiedBy(identifier) )
                return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    std::vector< osg::observer_ptr<GLMemory> >& GLMemoryTargetCallback::getObservedMemoryList()
    {
        return _memories;
    }

    //------------------------------------------------------------------------------
    const std::vector< osg::observer_ptr<GLMemory> >& GLMemoryTargetCallback::getObservedMemoryList() const
    {
        return _memories;
    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::operator()( osg::RenderInfo& renderInfo ) const
    {
        for( std::vector< osg::observer_ptr<GLMemory> >::const_iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if((*itr).valid()) (*itr)->mapAsRenderTarget();
        }
    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::operator()( const osg::Camera& camera ) const
    {
        for( std::vector< osg::observer_ptr<GLMemory> >::const_iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if((*itr).valid()) (*itr)->mapAsRenderTarget();
        }
    }

    //------------------------------------------------------------------------------
    void GLMemoryTargetCallback::drawImplementation( osg::RenderInfo& renderInfo,const osg::Drawable* drawable ) const
    {
        for( std::vector< osg::observer_ptr<GLMemory> >::const_iterator itr = _memories.begin(); itr != _memories.end(); ++itr )
        {
            if((*itr).valid()) (*itr)->mapAsRenderTarget();
        }

        drawable->drawImplementation(renderInfo);
    }

} 
