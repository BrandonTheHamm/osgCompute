#include <osg/NodeVisitor>
#include <osg/Geode>
#include <osg/Texture>
#include <osg/Group>
#include <osgCudaUtil/PingPongSwitch>

namespace osgCuda
{

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // STATIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    osg::Switch* PingPongSwitch::createPingPongGeodeSwitch( osgCuda::PingPongBuffer* pingPongBuffer )
    {
        if( pingPongBuffer == NULL )
            return NULL;

        osgCuda::PingPongSwitch* ppSwitch = new osgCuda::PingPongSwitch;
        ppSwitch->setPingPongBuffer( pingPongBuffer );

        for( unsigned int g=0; g< pingPongBuffer->getSwapCount(); ++g )
        {
            osgCompute::GLMemory* glMemory = dynamic_cast<osgCompute::GLMemory*>(pingPongBuffer->getBufferAt(g));
            if( NULL == glMemory )
            {
                osg::notify(osg::WARN)
                    <<"PingPongSwitch::createPingPongTextureSwitch(): Cannot create switch. Buffer "
                    << pingPongBuffer->getName() << " at " 
                    << g << " is not a osgCompute::GLMemory object."
                    << std::endl;

                ppSwitch->unref();
                return NULL;
            }

            osg::Drawable* drawable = dynamic_cast<osg::Drawable*>( glMemory->getAdapter() );
            if( NULL == drawable ) 
            {
                osg::notify(osg::WARN)
                    <<"PingPongSwitch::createPingPongGeodeSwitch(): Cannot create switch. Buffer "
                    << pingPongBuffer->getName() << " at " 
                    << g << " is not a osg::Drawable object."
                    << std::endl;

                ppSwitch->unref();
                return NULL;
            }

            osg::ref_ptr<osg::Geode> geode = new osg::Geode;
            geode->addDrawable( drawable );
            ppSwitch->addChild( geode );
        }

        return ppSwitch;
    }

    //------------------------------------------------------------------------------
    osg::Switch* PingPongSwitch::createPingPongTextureSwitch( unsigned int unit, osgCuda::PingPongBuffer* pingPongBuffer )
    {
        if( pingPongBuffer == NULL )
            return NULL;

        osgCuda::PingPongSwitch* ppSwitch = new osgCuda::PingPongSwitch;
        ppSwitch->setPingPongBuffer( pingPongBuffer );

        for( unsigned int g=0; g< pingPongBuffer->getSwapCount(); ++g )
        {
            osgCompute::GLMemory* glMemory = dynamic_cast<osgCompute::GLMemory*>(pingPongBuffer->getBufferAt(g));
            if( NULL == glMemory )
            {
                osg::notify(osg::WARN)
                    <<"PingPongSwitch::createPingPongTextureSwitch(): Cannot create switch. Buffer "
                    << pingPongBuffer->getName() << " at " 
                    << g << " is not a osgCompute::GLMemory object."
                    << std::endl;

                ppSwitch->unref();
                return NULL;
            }

            osg::Texture* texture = dynamic_cast<osg::Texture*>( glMemory->getAdapter() );
            if( NULL == texture ) 
            {
                osg::notify(osg::WARN)
                    <<"PingPongSwitch::createPingPongTextureSwitch(): Cannot create switch. Buffer "
                    << pingPongBuffer->getName() << " at " 
                    << g << " is not a osgCuda::TextureXD object."
                    << std::endl;

                ppSwitch->unref();
                return NULL;
            }

            osg::ref_ptr<osg::Group> group = new osg::Group;
            group->getOrCreateStateSet()->setTextureAttribute( unit, texture, osg::StateAttribute::ON );
            ppSwitch->addChild( group );
        }

        return ppSwitch;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    PingPongSwitch::PingPongSwitch()
        : osg::Switch()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void PingPongSwitch::accept( osg::NodeVisitor& nv )
    {
        if( nv.getVisitorType() == osg::NodeVisitor::CULL_VISITOR )
        {
            osg::Switch::setSingleChildOn( _pingPongBuffer->getSwapIdx() );
        }

        osg::Switch::accept( nv );
    }

    //------------------------------------------------------------------------------
    bool PingPongSwitch::setPingPongBuffer( osgCuda::PingPongBuffer* pingPongBuffer )
    {
        _pingPongBuffer = pingPongBuffer;
        return true;
    }

    //------------------------------------------------------------------------------
    osgCuda::PingPongBuffer* PingPongSwitch::getPingPongBuffer()
    {
        return _pingPongBuffer.get();
    }

    //------------------------------------------------------------------------------
    const osgCuda::PingPongBuffer* PingPongSwitch::getPingPongBuffer() const
    {
        return _pingPongBuffer.get();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PingPongSwitch::clearLocal()
    {
        _pingPongBuffer = NULL;
    }


}