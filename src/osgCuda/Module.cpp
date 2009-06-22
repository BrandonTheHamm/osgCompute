#include <osgCuda/Module>

namespace osgCuda
{   
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC FUNCTIONS /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    Module::Module() 
        : osgCompute::Module() 
    { 
        clearLocal(); 
    }

    //------------------------------------------------------------------------------
    bool Module::init() 
    { 
        return osgCompute::Module::init(); 
    }

    //------------------------------------------------------------------------------
    void Module::clear()
    {
        clearLocal(); 
        osgCompute::Module::clear();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS //////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void Module::clearLocal()
    {
    }

    //------------------------------------------------------------------------------
    bool Module::init( const osgCompute::Context& context ) const
    {
        return osgCompute::Module::init( context );
    }

    //------------------------------------------------------------------------------
    void Module::clear( const osgCompute::Context& context ) const
    {
        osgCompute::Module::clear( context );
    }
}
