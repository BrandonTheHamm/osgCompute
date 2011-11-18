#include <osgCudaUtil/Timer>

namespace osgCuda
{   
    bool Timer::_timerEnabled = true;
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Timer::enableAllTimer()
    {
        _timerEnabled = true;
    }

    //------------------------------------------------------------------------------
    void Timer::disableAllTimer()
    {
        _timerEnabled = false;
    }

    //------------------------------------------------------------------------------
    bool Timer::timerEnabled()
    {
        return _timerEnabled;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Timer::Timer() : 
        _start(NULL),
        _stop(NULL),
        _calls(0),
        _lastTime(0.0f),
        _peakTime(0.0f),
        _overallTime(0.0f)
    {
        clearLocal();
        // Please note that virtual functions className() and libraryName() are called
        // during observeResource() which will only develop until this class.
        // However if contructor of a subclass calls this function again observeResource
        // will change the className and libraryName of the observed pointer.
        osgCompute::ResourceObserver::instance()->observeResource( *this );
    }

    //------------------------------------------------------------------------------
    Timer::~Timer()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    void Timer::start()
    {
        if( !timerEnabled() )
            return;

        if( _start == NULL )
            cudaEventCreate(&_start); 

        cudaEventRecord( _start );
    }

    //------------------------------------------------------------------------------
    void Timer::stop()
    {
        if( !timerEnabled() )
            return;

        if( _stop == NULL )
            cudaEventCreate(&_stop);

        cudaEventRecord( _stop );

        cudaEventSynchronize( _stop ); 
        cudaEventElapsedTime(&_lastTime, _start, _stop);

        if( _peakTime < _lastTime )
            _peakTime = _lastTime;

        _overallTime += _lastTime;
        _calls++;
    }

    //------------------------------------------------------------------------------
    float Timer::getAveTime() const
    {
        if( _overallTime == 0.0f || _calls == 0 )
            return 0.0f;
        else
            return _overallTime / static_cast<float>( _calls );
    }

    //------------------------------------------------------------------------------
    float Timer::getLastTime() const
    {
        return _lastTime;
    }

    //------------------------------------------------------------------------------
    float Timer::getPeakTime() const
    {
        return _peakTime;
    }

    //------------------------------------------------------------------------------
    unsigned int Timer::getCalls() const
    {
        return _calls;
    }

    //------------------------------------------------------------------------------
    void Timer::releaseObjects()
    {
        if( _start != NULL )
            cudaEventDestroy(_start); 
        _start = NULL;

        if( _stop != NULL )
            cudaEventDestroy(_stop);
        _stop = NULL;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Timer::clearLocal()
    {
        _lastTime = 0.0f;
        _peakTime = 0.0f;
        _overallTime = 0.0f;
        _calls = 0;
    }
}
