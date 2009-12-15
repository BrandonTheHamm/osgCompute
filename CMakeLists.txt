################################################################################
# Set cmake version and do some general work (set project, CMAKE_MODULE_PATH,...)
################################################################################

cmake_minimum_required(VERSION 2.6)

PROJECT(osgCompute)

# add modules to cmake module path
############################
# We have some custom .cmake scripts not in the official distribution.
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${osgCompute_SOURCE_DIR}/CMakeModules")

# Make the comfortable  cmake modules of CMake 2.6.3 also available for users of older
# cmake versions instead of requiring CMake >= 2.6.3
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${osgCompute_SOURCE_DIR}/CMakeModules/osg")

# set custom .cmake CUDA scripts also to the module path
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${osgCompute_SOURCE_DIR}/CMakeModules/cuda")



################################################################################
# Find some necessary packages
################################################################################

# Find OpenGL
FIND_PACKAGE(OpenGL)



################################################################################
# Set basic unclude directories: i.e. show some headers to everyone
################################################################################

# Make the headers visible to everything
IF(NOT ${PROJECT_BINARY_DIR} EQUAL ${PROJECT_SOURCE_DIR})
   INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)
ENDIF(NOT ${PROJECT_BINARY_DIR} EQUAL ${PROJECT_SOURCE_DIR})


INCLUDE_DIRECTORIES(
    ${osgCompute_SOURCE_DIR}/include
    # do this for general use, because almost every lib may need to link against opengl
    ${OPENGL_INCLUDE_DIR}
)



################################################################################
# Platform specific definitions
################################################################################

IF(WIN32)
    IF(MSVC)
        ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)
        ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
    ENDIF(MSVC)
ENDIF(WIN32)



################################################################################
# check for necessary libraries
################################################################################

# check for some libs and scrips.
INCLUDE(Findosg)    # sets OSG_FOUND
INCLUDE(FindCuda)   # sets CUDA_FOUND



################################################################################
# Create bin and lib directories in build path
################################################################################

# Create bin and lib directories if required
IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin ${CMAKE_BINARY_DIR}/lib)
ENDIF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")



################################################################################
# Installation stuff
################################################################################

SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
ADD_DEFINITIONS(-DDEBUG_POSTFIX="${CMAKE_DEBUG_POSTFIX}")
IF(UNIX AND NOT WIN32 AND NOT APPLE)
  IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ADD_DEFINITIONS("-D_DEBUG")
  ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")
ENDIF(UNIX AND NOT WIN32 AND NOT APPLE)


# Check for 64 Bit system 
IF(UNIX AND NOT WIN32 AND NOT APPLE)
  IF(CMAKE_SIZEOF_VOID_P MATCHES "8")
      SET(LIB_POSTFIX "64" CACHE STRING "suffix for 32/64 dir placement")
      MARK_AS_ADVANCED(LIB_POSTFIX)
  ENDIF(CMAKE_SIZEOF_VOID_P MATCHES "8")
ENDIF(UNIX AND NOT WIN32 AND NOT APPLE)
IF(NOT DEFINED LIB_POSTFIX)
    SET(LIB_POSTFIX "")
ENDIF(NOT DEFINED LIB_POSTFIX)


# Make the bin/ and lib/ folders
SET(OUTPUT_BINDIR ${PROJECT_BINARY_DIR}/bin)
MAKE_DIRECTORY(${OUTPUT_BINDIR})

SET(OUTPUT_LIBDIR ${PROJECT_BINARY_DIR}/lib)
MAKE_DIRECTORY(${OUTPUT_LIBDIR})


# On CMake 2.6.x use the newly minted CMAKE_LIBRARY_OUTPUT_DIRECTORY,
# CMAKE_ARCHIVE_OUTPUT_DIRECTORY & CMAKE_RUNTIME_OUTPUT_DIRECTORY
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_LIBDIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINDIR})
IF(WIN32)
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_BINDIR})
ELSE(WIN32)
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_LIBDIR})
ENDIF(WIN32)



################################################################################
# User Options
################################################################################

############################
# Dynamic vs Static Linking
############################
OPTION(DYNAMIC_LINKING "Set to ON to build the module for dynamic linking.  Use OFF for static." ON)
IF   (DYNAMIC_LINKING)
    SET(LINKING_USER_DEFINED_DYNAMIC_OR_STATIC "SHARED")
ELSE (DYNAMIC_LINKING)
    SET(LINKING_USER_DEFINED_DYNAMIC_OR_STATIC "STATIC")
ENDIF(DYNAMIC_LINKING)

############################
# Include important macro
############################
INCLUDE(MacroUtils)

############################
# add sources
############################
ADD_SUBDIRECTORY(src)


############################
# Build multithreading version - not supported at the moment
############################
#OPTION(BUILD_MULTITHREADING_VERSION "Enable to build osgCompute and osgCuda with osg multi-threading support" ON)


############################
# Examples
############################
OPTION(BUILD_EXAMPLES "Enable to build Examples" ON)
IF   (BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF(BUILD_EXAMPLES)



################################################################################
# Uninstall target
## - check if it also works under linux
################################################################################

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
  