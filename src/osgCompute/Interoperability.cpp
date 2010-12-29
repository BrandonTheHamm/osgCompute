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
#include <osg/GraphicsContext>
#include <osgCompute/Interoperability>

namespace osgCompute
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void InteropObject::setUsage( unsigned int usage )
    {
        _usage = usage;
    }

    //------------------------------------------------------------------------------
    unsigned int InteropObject::getUsage() const
    {
        return _usage;
    }

    //------------------------------------------------------------------------------
    void InteropObject::clear()
    {
        clearLocal();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void InteropObject::clearLocal()
    {
        _usage = GL_SOURCE_COMPUTE_SOURCE;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void InteropMemory::clear()
    {
        clearLocal();
        osgCompute::Memory::clear();
    }

    //------------------------------------------------------------------------------
    void InteropMemory::clearObject()
    {
        if( Resource::getContextID() != UINT_MAX )
        {
            osg::GraphicsContext::GraphicsContexts contexts = osg::GraphicsContext::getRegisteredGraphicsContexts(Resource::getContextID());
            if( !contexts.empty() && contexts.front()->isRealized() )
            {      
                // Make context the current context
                if( !contexts.front()->isCurrent() )
                    contexts.front()->makeCurrent();
            }
            else if( contexts.empty() )
            {
                osg::notify(osg::FATAL) 
                    << "[InteropMemory::clearObject()]: "
                    << "the associated graphics context is not available anymore."
                    << "Check that you call releaseGLObjects(state) before removing the context."
                    << "Maybe freeing OpenGL related resources is not possible."
                    << std::endl;
            }
        }

        osgCompute::Memory::clearObject();
    }
        
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void InteropMemory::clearLocal()
    {
        //InteropMemory::clearObject() is called by Resource::clear()
    }
}
