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

#include <sstream>
#include <osg/Notify>
#include "osgCompute/Context"

namespace osgCompute
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    typedef std::map< std::string, osg::ref_ptr<Context> >                  ContextProtoMap;
    typedef std::map< std::string, osg::ref_ptr<Context> >::iterator        ContextProtoMapItr;
    typedef std::map< std::string, osg::ref_ptr<Context> >::const_iterator  ContextProtoMapCnstItr;

    static OpenThreads::Mutex s_contextMutex;
    static ContextMap s_contexts;
    static ContextProtoMap s_contextProtos;

    //------------------------------------------------------------------------------
    Context* Context::instance( unsigned int id, bool erase /*= false*/ )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_contextMutex);

        Context* context = NULL;

        ContextMapItr itr = s_contexts.find( id );
        if( itr != s_contexts.end() )
        {
            context = (*itr).second.get();

            if( erase )
            {
                s_contexts.erase( itr );

                if( context )
                {
                    delete context;
                    context = NULL;
                }
            }
        }

        return context;
    }

    //------------------------------------------------------------------------------
    Context* Context::instance( osg::State& state, bool erase /*= false*/ )
    {
        return Context::instance( state.getContextID(), erase );
    }

    //------------------------------------------------------------------------------
    Context* Context::createInstance( unsigned int id, std::string libraryName, std::string className )
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_contextMutex);

        Context* context = NULL;

        ContextMapItr itr = s_contexts.find( id );
        if( itr != s_contexts.end() )
        {
            context = (*itr).second.get();
        }
        else
        {
            std::string contextSpec = libraryName;
            contextSpec.append("::");
            contextSpec.append( className );

            ContextProtoMapItr protoItr = s_contextProtos.find( contextSpec );
            if( protoItr != s_contextProtos.end() )
            {
                context = dynamic_cast<Context*>( (*protoItr).second->cloneType() );
                if( context )
                {
                    context->setId( id );

                    std::stringstream contextName;
                    contextName << "Context " << id; 
                    context->setName( contextName.str() );
                    s_contexts.insert( std::make_pair< unsigned int, osg::ref_ptr<Context> >( id, context) );
                }
                else
                {
                    osg::notify(osg::FATAL)  << "Context::createContext(): cannot create context of type \""
                        << contextSpec << "\"."
                        << std::endl;
                }
            }
            else
            {
                osg::notify(osg::FATAL)  << "Context::createContext(): cannot find a context of type \""
                    << contextSpec << "\"."
                    << std::endl;
            }

        }

        return context;
    }

    //------------------------------------------------------------------------------
    Context* Context::createInstance( osg::State& state, std::string libraryName, std::string className )
    {
        Context* context = Context::createInstance( state.getContextID(), libraryName, className );

        if( context )
            context->setState( state );

        return context;
    }

    //------------------------------------------------------------------------------
    void Context::registerContext( Context& contextProto )
    {
        std::string contextSpec = contextProto.libraryName();
        contextSpec.append("::");
        contextSpec.append( contextProto.className() );

        ContextProtoMapItr protoItr = s_contextProtos.find( contextSpec );
        if( protoItr == s_contextProtos.end() )
        {
            s_contextProtos.insert( std::make_pair< std::string, osg::ref_ptr<Context> >( contextSpec, &contextProto) );
        }
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ContextResource::ContextResource()
        : osg::Object()
    {
    }

    //------------------------------------------------------------------------------
    void ContextResource::clear()
    {
        clearLocal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    ContextResource::~ContextResource()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void ContextResource::clearLocal()
    {
        for( ContextSetItr itr = _contexts.begin(); itr != _contexts.end(); ++itr )
        {
            Context* ctx = Context::instance( (*itr) );
            if( ctx != NULL && 
                ctx->isRegistered( const_cast<ContextResource&>(*this) ) )
                ctx->unregisterResource( const_cast<ContextResource&>(*this) );
        }
        _contexts.clear();

        osg::Object::setUserData( NULL );
    }

    //------------------------------------------------------------------------------
    bool ContextResource::init( const Context& context ) const
    {
        if( context.isRegistered(const_cast<ContextResource&>(*this)) )
            return true;

        context.registerResource(const_cast<ContextResource&>(*this));

        ContextSetItr itr = _contexts.find( context.getId() );
        if( itr == _contexts.end() )
            _contexts.insert( context.getId() );

        return true;
    }

    //------------------------------------------------------------------------------
    void ContextResource::clear( const Context& context ) const
    {
        context.unregisterResource(const_cast<ContextResource&>(*this));

        ContextSetItr itr = _contexts.find( context.getId() );
        if( itr != _contexts.end() )
            _contexts.erase( itr );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Context::Context()
        : osg::Object()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    Context::~Context()
    {
        clearLocal();
    }


    //------------------------------------------------------------------------------
    bool Context::init()
    {
        _dirty = false;
        return true;
    }

    //------------------------------------------------------------------------------
    void Context::apply()
    {
        if(isDirty())
            init();
    }

    //------------------------------------------------------------------------------
    void Context::clear()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool osgCompute::Context::isRegistered( ContextResource& resource ) const
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        for( ResourceListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
            if( (*itr) == &resource )
                return true;

        return false;
    }

    //------------------------------------------------------------------------------
    void Context::registerResource( ContextResource& resource ) const
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        _resources.push_back( &resource );
    }

    //------------------------------------------------------------------------------
    void Context::unregisterResource( ContextResource& resource ) const
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        for( ResourceListItr itr = _resources.begin(); itr != _resources.end(); ++itr )
        {
            if( (*itr) == &resource )
            {
                _resources.erase( itr );
                return;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Context::clearLocal()
    {
        // free context dependent memory
        clearResources();

        // do not clear the id!!!
        _state = NULL;
        _dirty = true;
    }

    //------------------------------------------------------------------------------
    void Context::clearResources() const
    {
        while( !_resources.empty() )
        {
            ContextResource* curResource = _resources.front();

            // each of the resources calls unregisterParam()
            // within the clear(\"CONTEXT\") function
            if( curResource )
                curResource->clear( *this );
        }
    }
}
