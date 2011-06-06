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
    }

    //------------------------------------------------------------------------------
    Timer::~Timer()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool Timer::init()
    {
        cudaEventCreate(&_start); 
        cudaEventCreate(&_stop);

        return Resource::init();
    }

    //------------------------------------------------------------------------------
    void Timer::start()
    {
        if( !timerEnabled() )
            return;

        if( isClear() )
            if( !init() )
                return;

        cudaEventRecord( _start );
    }

    //------------------------------------------------------------------------------
    void Timer::stop()
    {
        if( isClear() || !timerEnabled() )
            return;

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
    void Timer::clear()
    {
        clearLocal();
        Resource::clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Timer::clearLocal()
    {
        if( _start != NULL )
            cudaEventDestroy(_start); 
        _start = NULL;
        
        if( _stop != NULL )
            cudaEventDestroy(_stop);
        _stop = NULL    ;

        _lastTime = 0.0f;
        _peakTime = 0.0f;
        _overallTime = 0.0f;
        _calls = 0;
    }
}
