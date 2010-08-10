#include "Util.h"

namespace osgCuda
{
    //------------------------------------------------------------------------------
    std::string trim( const std::string& str )
    {
        if (!str.size()) return str;
        std::string::size_type first = str.find_first_not_of( " \t" );
        std::string::size_type last = str.find_last_not_of( "  \t\r\n" );
        if ((first==str.npos) || (last==str.npos)) return std::string( "" );
        return str.substr( first, last-first+1 );
    }
}