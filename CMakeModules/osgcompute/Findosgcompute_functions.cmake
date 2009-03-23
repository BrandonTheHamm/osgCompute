#
# This CMake file contains two macros to assist with searching for osgCompute
# libraries. This file is originally privided by cmake/osg community and has been adapted
# to osgPipe by the SVT Group. 
#
# NOTE: the HINTS paths seem not being searched in CMake 2.6 (-SVT Group)
# Therefore we use effectively just the paths in PATHS
#

function(OSGCOMPUTE_FIND_PATH module header)
   string(TOUPPER ${module} module_uc)
   
   # Try the user's environment request before anything else.
   find_path(${module_uc}_INCLUDE_DIR ${header}
       HINTS
            ENV ${module_uc}_DIR
            ENV OSGCOMPUTE_DIR
            ENV OSGCOMPUTEDIR
       PATH_SUFFIXES include
       PATHS
            ENV ${module_uc}_DIR
            ENV OSGCOMPUTE_DIR
            ENV OSGCOMPUTEDIR
            ~/Library/Frameworks
            /Library/Frameworks
            /sw # Fink
            /opt/local # DarwinPorts
            /opt/csw # Blastwave
            /opt
            [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OSGCOMPUTEDIR]
   )
   
endfunction(OSGCOMPUTE_FIND_PATH module header)

function(OSGCOMPUTE_FIND_LIBRARY module library)
   string(TOUPPER ${module} module_uc)

   find_library(${module_uc}_LIBRARY
       NAMES ${library}
       HINTS
            ENV ${module_uc}_DIR
            ENV OSGCOMPUTE_DIR
            ENV OSGCOMPUTEDIR
       PATH_SUFFIXES lib64 lib
       PATHS
            ENV ${module_uc}_DIR
            ENV OSGCOMPUTE_DIR
            ENV OSGCOMPUTEDIR
            ~/Library/Frameworks
            /Library/Frameworks
            /usr/local
            /usr
            /sw # Fink
            /opt/local # DarwinPorts
            /opt/csw # Blastwave
            /opt
            [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OSGCOMPUTEDIR]
   )

   if(MSVC)
       # When compiling with VS, search for debug libraries since they are
       # nearly always needed at runtime. 
       find_library(${module_uc}_LIBRARY_DEBUG
           NAMES ${library}d
           HINTS
                ENV ${module_uc}_DIR
                ENV OSGCOMPUTE_DIR
                ENV OSGCOMPUTEDIR
           PATH_SUFFIXES lib64 lib
           PATHS
                ENV ${module_uc}_DIR
                ENV OSGCOMPUTE_DIR
                ENV OSGCOMPUTEDIR
                ~/Library/Frameworks
                /Library/Frameworks
                /usr/local
                /usr
                /sw # Fink
                /opt/local # DarwinPorts
                /opt/csw # Blastwave
                /opt
                [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OSGCOMPUTEDIR]
       )
   else(MSVC)
       #
       # On all other platforms there is no requirement to link
       # debug targets against debug libraries and release targets
       # against release libraries so just set the FOO_LIBRARY_DEBUG
       # for the users' convenience in calling target_link_libraries()
       # once.
       #
       set(${module_uc}_LIBRARY_DEBUG ${${module_uc}_LIBRARY})
   endif(MSVC)

endfunction(OSGCOMPUTE_FIND_LIBRARY module library)
