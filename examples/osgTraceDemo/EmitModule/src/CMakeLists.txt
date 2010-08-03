#########################################################################
# ATTENTION: THIS LIB IS A PLUGIN
#########################################################################

#########################################################################
# Set library and plugin name
#########################################################################

SET(LIB_NAME osgcuda_ptclemitter)
SET(LIB_NAME_LABEL "Module ${LIB_NAME}")


#########################################################################
# Do necessary checking stuff
#########################################################################

INCLUDE(FindOpenThreads)
INCLUDE(Findosg)
INCLUDE(FindosgDB)
INCLUDE(Findmatlab)
INCLUDE(FindCuda)

#########################################################################
# Set path to header files
#########################################################################

SET(HEADER_PATH ${PROJECT_SOURCE_DIR}/include/${MODULE_NAME})


#########################################################################
# Set basic include directories
#########################################################################

INCLUDE_DIRECTORIES(
    ${HEADER_PATH}
	${OSG_INCLUDE_DIR}
	${CUDA_INCLUDE_PATH}
)





#########################################################################
# Collect header and source files
#########################################################################

# collect all headers

SET(TARGET_H
)

SET(MY_CUDA_SOURCE_FILES
	PtclEmitter.cu
)

# Use the WRAP_CUDA macro.
WRAP_CUDA( CUDA_FILES ${MY_CUDA_SOURCE_FILES} )

# collect the sources
SET(TARGET_SRC
	PtclEmitter.cpp
	${CUDA_FILES}
    ${MY_CUDA_SOURCE_FILES} 
)


#########################################################################
# Setup groups for resources (mainly for MSVC project folders)
#########################################################################

# Setup groups for headers (especially for files with no extension)
SOURCE_GROUP(
    "Header Files"
    FILES ${TARGET_H}     
)

# Setup groups for sources 
SOURCE_GROUP(
    "Source Files"
    FILES ${TARGET_SRC}
)

# finally, use module to build groups
INCLUDE(GroupInstall)

#########################################################################
# Build Library and prepare install scripts
#########################################################################

IF(DYNAMIC_LINKING)
    ADD_LIBRARY(${LIB_NAME} MODULE ${TARGET_SRC} ${TARGET_H})
ELSE (DYNAMIC_LINKING)
    ADD_LIBRARY(${LIB_NAME} STATIC ${TARGET_SRC} ${TARGET_H})
ENDIF(DYNAMIC_LINKING)

# install the module to the executable of the application 
INSTALL(TARGETS ${LIB_NAME} LIBRARY DESTINATION share/bin)


SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES PROJECT_LABEL "${LIB_NAME_LABEL}")

#########################################################################
# Linking
#########################################################################

# link here the project libraries    
TARGET_LINK_LIBRARIES(${LIB_NAME}
    osgCompute
	osgCuda
)


# use this macro for linking with libraries that come from Findxxxx commands
# this adds automatically "optimized" and "debug" information for cmake 
LINK_WITH_VARIABLES(${LIB_NAME}
	OPENTHREADS_LIBRARY
	OSG_LIBRARY
	OSGDB_LIBRARY
    CUDA_RUNTIME_LIBRARY
)
