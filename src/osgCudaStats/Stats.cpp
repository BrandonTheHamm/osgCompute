#include <sstream>
#include <osg/PolygonMode>
#include <osg/Uniform>
#include <osg/ApplicationUsage>
#include <osg/Geometry>
#include <osgText/Text>
#include <osgViewer/Renderer>
#include <osgCompute/Resource>
#include <osgCompute/Memory>
#include <osgCudaStats/Stats>

namespace osgCuda
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // CALLBACK FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /**
    */
    struct TimeBlockCallback : public osg::Drawable::DrawCallback
    {
    public:
        enum Type
        {
            LAST_TIME = 0,
            AVE_TIME = 1,
            PEAK_TIME = 2
        };

    public:
        //------------------------------------------------------------------------------
        TimeBlockCallback( osgCompute::ResourceClassList& timerList, float blockStart, float maxBlockLength, Type type )
            : _maxBlockLength(maxBlockLength),
			  _blockStart(blockStart),
              _type( type )
        {
            _timerList = timerList;
        }

        //------------------------------------------------------------------------------
        virtual void drawImplementation(osg::RenderInfo& renderInfo ,const osg::Drawable* drawable) const
        {
            osg::Geometry* geom = (osg::Geometry*)drawable;
            osg::Vec3Array* vertices = (osg::Vec3Array*)geom->getVertexArray();

            float maxTime = 0.0f;
            for( osgCompute::ResourceClassListCnstItr itr = _timerList.begin(); itr != _timerList.end(); ++itr )
            {
                if( !(*itr).valid() ) 
                    continue;

                const osgCuda::Timer* curTimer = dynamic_cast<const osgCuda::Timer*>( (*itr).get() );
                if( NULL == curTimer )
                    continue;

                switch( _type )
                {
                case LAST_TIME:
                    maxTime = osg::maximum( curTimer->getLastTime(), maxTime );
                    break;
                case AVE_TIME:
                    maxTime = osg::maximum( curTimer->getAveTime(), maxTime );
                    break;
                case PEAK_TIME:
                    maxTime = osg::maximum( curTimer->getPeakTime(), maxTime );
                    break;
                }
            }

            unsigned int tv=0;
            for( osgCompute::ResourceClassListCnstItr itr = _timerList.begin(); itr != _timerList.end(); ++itr )
            {
                if( !(*itr).valid() ) 
                {
                    tv += 4;
                    continue;
                }

                const osgCuda::Timer* curTimer = dynamic_cast<const osgCuda::Timer*>( (*itr).get() );
                if( NULL == curTimer )
                {
                    tv += 4;
                    continue;
                }

                float curTimerBlockLength;
                switch( _type )
                {
                case LAST_TIME:
                    curTimerBlockLength = curTimer->getLastTime()/maxTime * _maxBlockLength;
                    break;
                case AVE_TIME:
                    curTimerBlockLength = curTimer->getAveTime()/maxTime * _maxBlockLength;
                    break;
                case PEAK_TIME:
                    curTimerBlockLength = curTimer->getPeakTime()/maxTime * _maxBlockLength;
                    break;
                default:    
                    curTimerBlockLength = 0.0f;
                    break;
                }

                (*vertices)[tv++].x() = _blockStart;
                (*vertices)[tv++].x() = _blockStart;
                (*vertices)[tv++].x() = _blockStart + curTimerBlockLength;
                (*vertices)[tv++].x() = _blockStart + curTimerBlockLength;
            }

            osg::DrawArrays* drawArrays = static_cast<osg::DrawArrays*>(geom->getPrimitiveSet(0));
            drawArrays->setCount(tv);
            drawable->drawImplementation(renderInfo);
        }

    private:
        osgCompute::ResourceClassList _timerList;
        float                         _maxBlockLength;
        float                         _blockStart;
        Type                          _type;
    }; 

    /**
    */
    struct TimeTextDrawCallback : public osg::Drawable::DrawCallback
    {
    public:
        enum Type
        {
            LAST_TIME = 0,
            AVE_TIME = 1,
            PEAK_TIME = 2
        };

        TimeTextDrawCallback( const osg::observer_ptr<osgCuda::Timer> timer, Type type )
            : _timer(timer), _type(type)
        {
        }

        /** do customized draw code.*/
        virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
        {
            osgText::Text* text = (osgText::Text*)drawable;
            if( !text || !_timer)
                return;

            std::stringstream curStream;
            switch( _type )
            {
            case LAST_TIME:
                curStream << _timer->getLastTime(); 
                break;
            case AVE_TIME:
                curStream << _timer->getAveTime(); 
                break;
            case PEAK_TIME:
                 curStream << _timer->getPeakTime();
                break;
            }

            curStream << " ms";
            text->setText( curStream.str() );
            text->drawImplementation( renderInfo );
        }

        const osg::observer_ptr<osgCuda::Timer>      _timer;
        Type                                         _type;
    };

    /**
    */
    struct NumCallsTextDrawCallback : public virtual osg::Drawable::DrawCallback
    {
        NumCallsTextDrawCallback( const osg::observer_ptr<osgCuda::Timer> timer )
            : _timer(timer)
        {
        }

        /** do customized draw code.*/
        virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
        {
            osgText::Text* text = (osgText::Text*)drawable;
            if( !text || !_timer)
                return;

            std::stringstream curStream;
            curStream << "for " << _timer->getCalls() << " #calls";
            text->setText( curStream.str() );
            text->drawImplementation( renderInfo );
        }

         const osg::observer_ptr<osgCuda::Timer> _timer;
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    StatsHandler::StatsHandler() 
    {
        _statsType = STATISTICS_NONE;
        _hudCamera = NULL;
        _memorySwitch = NULL;
        _statsWidth = 1280.0f;
        _statsHeight = 1024.0f;
        osg::ApplicationUsage::instance()->addUsageExplanation(osg::ApplicationUsage::KEYBOARD_MOUSE_BINDING,"c","On/Off Display CUDA stats");
    }

    //------------------------------------------------------------------------------
    bool StatsHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa)
    {
        osgViewer::View* myview = dynamic_cast<osgViewer::View*>(&aa);
        if (!myview) return false;

        osgViewer::ViewerBase* viewer = myview->getViewerBase();
        if (ea.getHandled()) return false;

        switch(ea.getEventType())
        {
        case(osgGA::GUIEventAdapter::KEYDOWN):
            {
                if( ea.getKey() == 'c' &&
                    ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN )
                {
                    if( !_hudCamera.valid() )
                    {
                        if( !createHUDCamera(viewer) )
                        {
                            osg::notify(osg::WARN)<<"osgCudaStats::StatsHandler::handle(): cannot create hud camera"<<std::endl;
                            return false;
                        }
                    }

                    _statsType = (_statsType + 1) % STATISTICS_NUM_AVAILABLE;

                    // SWITCH CURRENT DISPLAYED STATUS
                    clearScene(viewer);
                    switch( _statsType )
                    {
                    case STATISTICS_NONE:
                        {
                            // disable all 
                            _hudCamera->setNodeMask(0x0);
                        }
                        break;
                    case STATISTICS_MEMORY:
                        {
                            setUpMemoryScene(viewer);
                            _memorySwitch->setSingleChildOn(0);
                            _hudCamera->setNodeMask(0xffffffff);
                        }
                        break;
                    case STATISTICS_TIMER:
                        {
                            setUpTimerScene(viewer);
                            _hudCamera->setNodeMask(0xffffffff);
                        }
                        break;

                    default: break;
                    }
                }
                if( /*ea.getKey() == osgGA::GUIEventAdapter::MODKEY_SHIFT &&*/ ea.getKey() == 'C' &&
                    ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN )
                {
                    if( _statsType == STATISTICS_MEMORY )
                    {
                        unsigned int curSwitchPage = 0;
                        for ( unsigned int i = 0; i < _memorySwitch->getNumChildren(); i++ )
                            if ( _memorySwitch->getValue(i) == true ) curSwitchPage = i;

                        curSwitchPage = ( (curSwitchPage + 1) % (_memorySwitch->getNumChildren()) );
                        // show next page if available
                        _memorySwitch->setSingleChildOn(curSwitchPage);
                    }

                } 
            }
        default: break;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    bool StatsHandler::createHUDCamera(osgViewer::ViewerBase* viewer)
    {
        osgViewer::ViewerBase::Contexts contexts;
        viewer->getContexts( contexts );
        osgViewer::GraphicsWindow* window = dynamic_cast<osgViewer::GraphicsWindow*>( contexts.front() );
        if (!window)
        {
            osg::notify(osg::WARN)<<"osgCudaStats::StatsHandler::createHUDCamera(): cannot find valid graphics context."<<std::endl;
            return false;
        }

        if( !_hudCamera.valid() ) _hudCamera = new osg::Camera;
        _hudCamera->setProjectionResizePolicy(osg::Camera::FIXED);
        _hudCamera->setGraphicsContext(window);
        _hudCamera->setViewport(0, 0, window->getTraits()->width, window->getTraits()->height);
        _hudCamera->setRenderOrder(osg::Camera::POST_RENDER, 10);
        _hudCamera->setProjectionMatrix(osg::Matrix::ortho2D(0.0,_statsWidth,0.0,_statsHeight));
        _hudCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        _hudCamera->setViewMatrix(osg::Matrix::identity());
        _hudCamera->setClearMask(0);
        _hudCamera->setRenderer(new osgViewer::Renderer(_hudCamera.get()));

        if( !_memorySwitch.valid() ) _memorySwitch = new osg::Switch;

        return true;
    }

    //------------------------------------------------------------------------------
    void StatsHandler::addModuleOfInterest( const std::string& moduleName )
    {
        std::set<std::string>::iterator itr = _modulesOfInterest.find( moduleName );
        if( itr != _modulesOfInterest.end() )
            return;

        _modulesOfInterest.insert( moduleName );
    }

    //------------------------------------------------------------------------------
    void StatsHandler::clearModulesOfInterest()
    {
        _modulesOfInterest.clear();
    }

    //------------------------------------------------------------------------------
    void StatsHandler::setUpMemoryScene(osgViewer::ViewerBase* viewer)
    {
        if( !_hudCamera.valid() )
            return;

        //osg::Geode* geode = new osg::Geode();
        //geode->setName("Memory");
        //_hudCamera->addChild( geode );
        _hudCamera->addChild( _memorySwitch );
        
        
        //osg::StateSet* stateset = geode->getOrCreateStateSet();
        osg::StateSet* stateset = _memorySwitch->getOrCreateStateSet();
        stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
        stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
        stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
        stateset->setAttribute(new osg::PolygonMode(), osg::StateAttribute::PROTECTED);

        std::string font("fonts/arial.ttf");

        unsigned int maxEntriesPerPage = 30;
        float leftPos = 10.0f;
        float characterSize = 18.0f;
        osg::Vec4 colorFR(1.0f,1.0f,1.0f,1.0f);

        osg::Vec3 posOrig(leftPos, _statsHeight-24.0f,0.0f);
        osg::Vec3 pos = posOrig;

        //////////////////////////////
        // COLLECT MEMORY RESOURCES //
        //////////////////////////////
        // Collect all memory consuming resources 
        osgCompute::ResourceClassList memoryList;
        osgCompute::ResourceClassList curResources;
        
        curResources = osgCompute::ResourceObserver::instance()->getResources( "osgCuda::Memory" );
        for( osgCompute::ResourceClassListItr itr = curResources.begin(); itr != curResources.end(); ++itr )
        {
            if( !(*itr).valid() || (NULL == dynamic_cast<const osgCompute::Memory*>((*itr).get())) ) 
                 continue;

            memoryList.push_back( (*itr) );
        }

        curResources = osgCompute::ResourceObserver::instance()->getResources( "osgCuda::TextureMemory" );
        for( osgCompute::ResourceClassListItr itr = curResources.begin(); itr != curResources.end(); ++itr )
        {
            if( !(*itr).valid() || (NULL == dynamic_cast<const osgCompute::Memory*>((*itr).get())) ) 
                continue;

            memoryList.push_back( (*itr) );
        }

        curResources = osgCompute::ResourceObserver::instance()->getResources( "osgCuda::GeometryMemory" );
        for( osgCompute::ResourceClassListItr itr = curResources.begin(); itr != curResources.end(); ++itr )
        {
            if( !(*itr).valid() || (NULL == dynamic_cast<const osgCompute::Memory*>((*itr).get())) ) 
                continue;

            memoryList.push_back( (*itr) );
        }

        curResources = osgCompute::ResourceObserver::instance()->getResources( "osgCuda::IndexedGeometryMemory" );
        for( osgCompute::ResourceClassListItr itr = curResources.begin(); itr != curResources.end(); ++itr )
        {
            if( !(*itr).valid() || (NULL == dynamic_cast<const osgCompute::Memory*>((*itr).get())) ) 
                continue;

            memoryList.push_back( (*itr) );
        }

        curResources = osgCompute::ResourceObserver::instance()->getResources( "osgCuda::PingPongBuffer" );
        for( osgCompute::ResourceClassListItr itr = curResources.begin(); itr != curResources.end(); ++itr )
        {
            if( !(*itr).valid() || (NULL == dynamic_cast<const osgCompute::Memory*>((*itr).get())) ) 
                continue;

            memoryList.push_back( (*itr) );
        }

        ///////////////////
        // Add Constants //
        ///////////////////
        float maxX = leftPos;
        // Compute max x pos
        for( osgCompute::ResourceClassListItr itr = memoryList.begin(); itr != memoryList.end(); ++itr )
        {
            const osgCompute::Memory* memory = dynamic_cast<const osgCompute::Memory*>((*itr).get());

            pos.x() = leftPos;
            osg::ref_ptr<osgText::Text> curLabel = new osgText::Text;
            curLabel->setColor(colorFR);
            curLabel->setFont(font);
            curLabel->setCharacterSize(characterSize);
            curLabel->setPosition(pos);
            if( memory->getName().empty() )
                curLabel->setText( "[NoName]" );
            else
                curLabel->setText( memory->getName() );

            float curX = curLabel->getBound().xMax();
            maxX = osg::maximum( curX, maxX );
        }

        float overallByteSize = 0.0f;
        unsigned int counter = 0;
        unsigned int pageCount = 1;
        osg::Geode* geode = NULL;
        unsigned int numPages = memoryList.size() / maxEntriesPerPage;
        if( memoryList.size() % maxEntriesPerPage != 0 )
            numPages++;

        // Add up all relevant constants
        for( osgCompute::ResourceClassListItr itr = memoryList.begin(); itr != memoryList.end(); ++itr )
        {
            
            // setup geode and page info if needed
            if ( counter % maxEntriesPerPage == 0 )
            {
                geode = new osg::Geode();
                geode->setName("Memory");
                _memorySwitch->addChild( geode );

                pos = posOrig;
                pos.x() = leftPos;

                osg::ref_ptr<osgText::Text> curPageInfo = new osgText::Text;
                geode->addDrawable( curPageInfo.get() );
                curPageInfo->setColor(colorFR);
                curPageInfo->setFont(font);
                curPageInfo->setCharacterSize(characterSize);
                curPageInfo->setPosition(pos);

                std::stringstream pagestream;
                pagestream << "Page " << pageCount << " of " << numPages << " (use SHIFT + c for showing next page):";
                curPageInfo->setText(pagestream.str());
                pageCount++;
                pos.y() -= characterSize*2.5f;
            }
            counter++;

            pos.x() = leftPos;

            const osgCompute::Memory* memory = dynamic_cast<const osgCompute::Memory*>((*itr).get());

            osg::ref_ptr<osgText::Text> curLabel = new osgText::Text;
            geode->addDrawable( curLabel.get() );
            osg::ref_ptr<osgText::Text> curValue = new osgText::Text;
            geode->addDrawable( curValue.get() );

            curLabel->setColor(colorFR);
            curLabel->setFont(font);
            curLabel->setCharacterSize(characterSize);
            curLabel->setPosition(pos);
            if( memory->getName().empty() )
                curLabel->setText( "[NoName]" );
            else
                curLabel->setText( memory->getName() );

            pos.x() = maxX;

            curValue->setColor(colorFR);
            curValue->setFont(font);
            curValue->setCharacterSize(characterSize);
            curValue->setPosition(pos);

            std::stringstream consstream;
            consstream.precision(4);
            consstream << " Dim=( "; 
            for( unsigned int d=0; d<memory->getNumDimensions(); ++d )
                    consstream <<  memory->getDimension(d)<< " ";
            consstream << ") "; 
            consstream << "Host= "<<memory->getMappingByteSize(osgCompute::MAP_HOST)/(1048576.0f)<<" MB; "; 
            consstream << "Device= "<<memory->getMappingByteSize(osgCompute::MAP_DEVICE)/(1048576.0f)<<" MB; "; 
            consstream << "Array= "<<memory->getMappingByteSize(osgCompute::MAP_DEVICE_ARRAY)/(1048576.0f)<<" MB; "; 
            consstream << "Sum= " << memory->getAllocatedByteSize()/(1048576.0f) << " MB";

            curValue->setText(consstream.str());
            pos.y() -= characterSize*1.5f;

            overallByteSize += memory->getAllocatedByteSize();
        }


         osg::ref_ptr<osgText::Text> overallLabel = new osgText::Text;
         geode->addDrawable( overallLabel.get() );
         pos.x() = leftPos;
         pos.y() -= characterSize*1.0f;

         std::stringstream consstream;
         consstream.precision(5);
         consstream << "Overall CUDA memory consumption = " << overallByteSize/(1048576.0f) << " MB";

         overallLabel->setColor(colorFR);
         overallLabel->setFont(font);
         overallLabel->setCharacterSize(characterSize);
         overallLabel->setPosition(pos);
         overallLabel->setText( consstream.str() );
         
    }

    //------------------------------------------------------------------------------
    void StatsHandler::setUpTimerScene(osgViewer::ViewerBase* viewer)
    {
        if( !_hudCamera.valid()  )
            return;

        osg::ref_ptr<osg::Geode> geode = new osg::Geode();
        geode->setName("Timer");
        _hudCamera->addChild( geode );

        osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();
        stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
        stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
        stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
        stateset->setAttribute(new osg::PolygonMode(), osg::StateAttribute::PROTECTED);

        float startX = 10.0f;
        float startY = _statsHeight-24.0f;
        osg::Vec3 pos(startX, startY,0.0f);
        std::string font("fonts/arial.ttf");
        float characterSize = 18.0f;
        osg::Vec4 colorFR(1.0f,1.0f,1.0f,1.0f);


        ////////////////////
        // COLLECT TIMERS //
        ////////////////////
        // Collect all memory consuming resources 
        osgCompute::ResourceClassList timerList = osgCompute::ResourceObserver::instance()->getResources( "osgCuda::Timer" );

        float curElementXPos = startX;

        float textElementSize = 180.0f;
        float blockElementSize = 120.0f;

        //////////////////////////////
        // Setup Table Header Texts //
        //////////////////////////////
        pos.x() = startX + 190.0f;
        float headerX = pos.x();

        osg::Vec4 colorHeader(1.0f,0.5f,0.5f,1.0f);

        osg::ref_ptr<osgText::Text> curLastTimeLabel = new osgText::Text;
        geode->addDrawable( curLastTimeLabel.get() );
        curLastTimeLabel->setColor(colorFR);
        curLastTimeLabel->setFont(font);
        curLastTimeLabel->setCharacterSize(characterSize+2.0f);
        curLastTimeLabel->setPosition(pos);
        curLastTimeLabel->setText( "Current Time" );

        headerX = curLastTimeLabel->getBound().xMax() + 160.0f;

        pos.x() = headerX;
        osg::ref_ptr<osgText::Text> curAveTimeLabel = new osgText::Text;
        geode->addDrawable( curAveTimeLabel.get() );
        curAveTimeLabel->setColor(colorFR);
        curAveTimeLabel->setFont(font);
        curAveTimeLabel->setCharacterSize(characterSize+2.0f);
        curAveTimeLabel->setPosition(pos);
        curAveTimeLabel->setText( "Average Time" );

        headerX = curAveTimeLabel->getBound().xMax() + 320.0f;

        pos.x() = headerX;
        osg::ref_ptr<osgText::Text> curPeakTimeLabel = new osgText::Text;
        geode->addDrawable( curPeakTimeLabel.get() );
        curPeakTimeLabel->setColor(colorFR);
        curPeakTimeLabel->setFont(font);
        curPeakTimeLabel->setCharacterSize(characterSize+2.0f);
        curPeakTimeLabel->setPosition(pos);
        curPeakTimeLabel->setText( "Peak Time" );

        startY -= (characterSize+2.0f) * 1.5f;


        /////////////////
        // Setup Texts //
        /////////////////
        pos.x() = startX;
        pos.y() = startY;
        // Compute max x pos
        for( osgCompute::ResourceClassListItr itr = timerList.begin(); itr != timerList.end(); ++itr )
        {
            const osgCuda::Timer* curTimer = dynamic_cast<const osgCuda::Timer*>((*itr).get());
            if( !(*itr).valid() || (NULL == curTimer) ) 
                continue;

            pos.x() = startX;
            osg::ref_ptr<osgText::Text> curLabel = new osgText::Text;
            geode->addDrawable( curLabel.get() );
            curLabel->setColor(colorFR);
            curLabel->setFont(font);
            curLabel->setCharacterSize(characterSize);
            curLabel->setPosition(pos);
            curLabel->setText( curTimer->getName() );

            float curX = curLabel->getBound().xMax();
            curElementXPos = osg::maximum( curX, curElementXPos );

            pos.y() -= characterSize*1.5f;
        }

        curElementXPos += 40.0f;

        curElementXPos =  (curElementXPos > 200.0f)? 200.0f : curElementXPos;

        //////////////////////////////
        // Add LAST Time Statistics //
        //////////////////////////////
        osg::ref_ptr<osg::Geometry> geometryLast = new osg::Geometry;
        geometryLast->setUseDisplayList(false);

        osg::ref_ptr<osg::Vec3Array> verticesLast = new osg::Vec3Array;
        geometryLast->setVertexArray(verticesLast);
        verticesLast->reserve(timerList.size()*4);

        // Initialize general positions
        pos.x() = curElementXPos;
        pos.y() = startY;
        for(unsigned int i=0; i<timerList.size(); ++i)
        {
            verticesLast->push_back(pos+osg::Vec3(0.0, characterSize, 0.0));
            verticesLast->push_back(pos+osg::Vec3(0.0, 0.0, 0.0));
            verticesLast->push_back(pos+osg::Vec3(10.0, 0.0, 0.0));
            verticesLast->push_back(pos+osg::Vec3(10.0, characterSize, 0.0));

            pos.y() -= characterSize*1.5f;
        }

        // Initialize color
        osg::ref_ptr<osg::Vec4Array> coloursLast = new osg::Vec4Array;
        coloursLast->push_back( osg::Vec4f(0.2f,0.9f,0.2f,1.0f) );
        geometryLast->setColorArray(coloursLast);
        geometryLast->setColorBinding(osg::Geometry::BIND_OVERALL);

        // Setup Quad Primitives
        geometryLast->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, timerList.size()*4));

        // Add callback
        float lastTimeBlockLength = blockElementSize;
        geometryLast->setDrawCallback( new TimeBlockCallback( timerList, curElementXPos, lastTimeBlockLength, TimeBlockCallback::LAST_TIME ) );
        curElementXPos += lastTimeBlockLength + 10.0f;

        // Attach blocks
        geode->addDrawable( geometryLast.get() );

        ////////////////////////
        // Add LAST Time Text //
        ////////////////////////
        pos.x() = curElementXPos;
        pos.y() = startY;
        // Add up all relevant constants
        for( osgCompute::ResourceClassListItr itr = timerList.begin(); itr != timerList.end(); ++itr )
        {
            osgCuda::Timer* curTimer = dynamic_cast<osgCuda::Timer*>((*itr).get());
            if( !(*itr).valid() || (NULL == curTimer) ) 
                continue;

            osg::ref_ptr<osgText::Text> curValue = new osgText::Text;
            geode->addDrawable( curValue.get() );

            curValue->setColor(colorFR);
            curValue->setFont(font);
            curValue->setCharacterSize(characterSize);
            curValue->setPosition(pos);
            curValue->setText("0.0");
            curValue->setDrawCallback( new TimeTextDrawCallback( curTimer, TimeTextDrawCallback::LAST_TIME ) );

            pos.y() -= characterSize*1.5f;
        }

        curElementXPos += textElementSize;

        /////////////////////////////
        // Add AVE Time Statistics //
        /////////////////////////////
        osg::ref_ptr<osg::Geometry> geometryAve = new osg::Geometry;
        geometryAve->setUseDisplayList(false);

        osg::ref_ptr<osg::Vec3Array> verticesAve = new osg::Vec3Array;
        geometryAve->setVertexArray(verticesAve);
        verticesAve->reserve(timerList.size()*4);

        // Initialize general positions
        pos.x() = curElementXPos;
        pos.y() = startY;
        for(unsigned int i=0; i<timerList.size(); ++i)
        {
            verticesAve->push_back(pos+osg::Vec3(0.0, characterSize, 0.0));
            verticesAve->push_back(pos+osg::Vec3(0.0, 0.0, 0.0));
            verticesAve->push_back(pos+osg::Vec3(10.0, 0.0, 0.0));
            verticesAve->push_back(pos+osg::Vec3(10.0, characterSize, 0.0));

            pos.y() -= characterSize*1.5f;
        }

        // Initialize color
        osg::ref_ptr<osg::Vec4Array> coloursAve = new osg::Vec4Array;
        coloursAve->push_back( osg::Vec4f(0.5f,0.5f,0.2f,1.0f) );
        geometryAve->setColorArray(coloursAve);
        geometryAve->setColorBinding(osg::Geometry::BIND_OVERALL);

        // Setup Quad Primitives
        geometryAve->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, timerList.size()*4));

        // Add callback
        float aveTimeBlockLength = blockElementSize;
        geometryAve->setDrawCallback( new TimeBlockCallback( timerList, curElementXPos, aveTimeBlockLength, TimeBlockCallback::AVE_TIME ) );
        curElementXPos += lastTimeBlockLength + 10.0f;

        // Attach blocks
        geode->addDrawable( geometryAve.get() );

        ////////////////////////
        // Add AVE Time Text //
        ////////////////////////
        pos.x() = curElementXPos;
        pos.y() = startY;
        // Add up all relevant constants
        for( osgCompute::ResourceClassListItr itr = timerList.begin(); itr != timerList.end(); ++itr )
        {
            osgCuda::Timer* curTimer = dynamic_cast<osgCuda::Timer*>((*itr).get());
            if( !(*itr).valid() || (NULL == curTimer) ) 
                continue;

            osg::ref_ptr<osgText::Text> curValue = new osgText::Text;
            geode->addDrawable( curValue.get() );

            curValue->setColor(colorFR);
            curValue->setFont(font);
            curValue->setCharacterSize(characterSize);
            curValue->setPosition(pos);
            curValue->setText("0.0");
            curValue->setDrawCallback( new TimeTextDrawCallback( curTimer, TimeTextDrawCallback::AVE_TIME  ) );

            pos.y() -= characterSize*1.5f;
        }

        curElementXPos += textElementSize - 30.0f;


        ////////////////////////
        // Add Num Calls Text //
        ////////////////////////
        pos.x() = curElementXPos;
        pos.y() = startY;
        // Add up all relevant constants
        for( osgCompute::ResourceClassListItr itr = timerList.begin(); itr != timerList.end(); ++itr )
        {
            osgCuda::Timer* curTimer = dynamic_cast<osgCuda::Timer*>((*itr).get());
            if( !(*itr).valid() || (NULL == curTimer) ) 
                continue;

            osg::ref_ptr<osgText::Text> curValue = new osgText::Text;
            geode->addDrawable( curValue.get() );

            curValue->setColor(colorFR);
            curValue->setFont(font);
            curValue->setCharacterSize(characterSize);
            curValue->setPosition(pos);
            curValue->setText("0");
            curValue->setDrawCallback( new NumCallsTextDrawCallback( curTimer ) );

            pos.y() -= characterSize*1.5f;
        }

        curElementXPos += textElementSize + 20.0f;

        /////////////////////////////
        // Add MAX Time Statistics //
        /////////////////////////////
        osg::ref_ptr<osg::Geometry> geometryPeak = new osg::Geometry;
        geometryPeak->setUseDisplayList(false);

        osg::ref_ptr<osg::Vec3Array> verticesPeak = new osg::Vec3Array;
        geometryPeak->setVertexArray(verticesPeak);
        verticesPeak->reserve(timerList.size()*4);

        // Initialize general positions
        pos.x() = curElementXPos;
        pos.y() = startY;
        for(unsigned int i=0; i<timerList.size(); ++i)
        {
            verticesPeak->push_back(pos+osg::Vec3(0.0, characterSize, 0.0));
            verticesPeak->push_back(pos+osg::Vec3(0.0, 0.0, 0.0));
            verticesPeak->push_back(pos+osg::Vec3(10.0, 0.0, 0.0));
            verticesPeak->push_back(pos+osg::Vec3(10.0, characterSize, 0.0));

            pos.y() -= characterSize*1.5f;
        }

        // Initialize color
        osg::ref_ptr<osg::Vec4Array> coloursPeak = new osg::Vec4Array;
        coloursPeak->push_back( osg::Vec4f(0.9f,0.2f,0.2f,1.0f) );
        geometryPeak->setColorArray(coloursPeak);
        geometryPeak->setColorBinding(osg::Geometry::BIND_OVERALL);

        // Setup Quad Primitives
        geometryPeak->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, timerList.size()*4));

        // Add callback
        float peakTimeBlockLength = blockElementSize;
        geometryPeak->setDrawCallback( new TimeBlockCallback( timerList, curElementXPos, peakTimeBlockLength, TimeBlockCallback::PEAK_TIME ) );
        curElementXPos += peakTimeBlockLength + 10.0f;

        // Attach blocks
        geode->addDrawable( geometryPeak.get() );


        ////////////////////////
        // Add Peak Time Text //
        ////////////////////////
        pos.x() = curElementXPos;
        pos.y() = startY;
        // Add up all relevant constants
        for( osgCompute::ResourceClassListItr itr = timerList.begin(); itr != timerList.end(); ++itr )
        {
            osgCuda::Timer* curTimer = dynamic_cast<osgCuda::Timer*>((*itr).get());
            if( !(*itr).valid() || (NULL == curTimer) ) 
                continue;

            osg::ref_ptr<osgText::Text> curValue = new osgText::Text;
            geode->addDrawable( curValue.get() );

            curValue->setColor(colorFR);
            curValue->setFont(font);
            curValue->setCharacterSize(characterSize);
            curValue->setPosition(pos);
            curValue->setText("0.0");
            curValue->setDrawCallback( new TimeTextDrawCallback( curTimer, TimeTextDrawCallback::PEAK_TIME ) );

            pos.y() -= characterSize*1.5f;
        }

        curElementXPos += textElementSize;
    }

    //------------------------------------------------------------------------------
    void StatsHandler::clearScene(osgViewer::ViewerBase* viewer)
    {
        _hudCamera->removeChildren(0, _hudCamera->getNumChildren() );
        _memorySwitch->removeChildren(0, _memorySwitch->getNumChildren() );
    }

}

