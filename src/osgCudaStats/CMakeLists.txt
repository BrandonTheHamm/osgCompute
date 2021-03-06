#########################################################################
# Set library name and set path to data folder of the library
#########################################################################

SET(LIB_NAME osgCudaStats)

IF(DYNAMIC_LINKING)
    ADD_DEFINITIONS(-DUSE_LIBRARY_DYN)
ELSE (DYNAMIC_LINKING)
    ADD_DEFINITIONS(-DUSE_LIBRARY_STATIC)
ENDIF(DYNAMIC_LINKING)


#########################################################################
# Do necessary checking stuff
#########################################################################

INCLUDE(FindOpenThreads)
INCLUDE(Findosg)
INCLUDE(FindosgDB)
INCLUDE(FindosgGA)
INCLUDE(FindosgText)
INCLUDE(FindCuda)


#########################################################################
# Set basic include directories
#########################################################################

INCLUDE_DIRECTORIES(
	${OSG_INCLUDE_DIR}
	${CUDA_TOOLKIT_INCLUDE}
)


#########################################################################
# Set path to header files
#########################################################################

SET(HEADER_PATH ${PROJECT_SOURCE_DIR}/include/${LIB_NAME})


#########################################################################
# Collect header and source files
#########################################################################

# collect all headers
SET(TARGET_H
    ${HEADER_PATH}/Stats
)


# collect the sources
SET(TARGET_SRC
	Stats.cpp
)


#########################################################################
# Setup groups for resources (mainly for MSVC project folders)
#########################################################################

# First: collect the necessary files which were not collected up to now
# Therefore, fill the following variables: 
# MY_ICE_FILES - MY_MODEL_FILES - MY_SHADER_FILES - MY_UI_FILES - MY_XML_FILES

# nothing todo so far in this module :-)

# finally, use module to build groups
#INCLUDE(GroupInstall)

# now set up the ADDITIONAL_FILES variable to ensure that the files will be visible in the project
SET(ADDITIONAL_FILES
#	${MY_ICE_FILES}
#	${MY_MODEL_FILES}
#	${MY_SHADER_FILES}
#	${MY_UI_FILES}
#	${MY_XML_FILES}
)


#########################################################################
# Build Library and prepare install scripts
#########################################################################

ADD_LIBRARY(${LIB_NAME}
    ${LINKING_USER_DEFINED_DYNAMIC_OR_STATIC}
	${TARGET_H}
    ${TARGET_SRC}
    ${ADDITIONAL_FILES}
)


# link here the project libraries    
TARGET_LINK_LIBRARIES(${LIB_NAME}
	osgCompute
    osgCuda
    osgCudaUtil
	#${OPENGL_LIBRARIES}
)

# use this macro for linking with libraries that come from Findxxxx commands
# this adds automatically "optimized" and "debug" information for cmake 
LINK_WITH_VARIABLES(${LIB_NAME}
	OPENTHREADS_LIBRARY
	OSG_LIBRARY
	OSGTEXT_LIBRARY
	OSGGA_LIBRARY
	OSGDB_LIBRARY
    CUDA_CUDART_LIBRARY
)

LINK_OPENGL_LIBRARIES(${LIB_NAME})

INCLUDE(ModuleInstall OPTIONAL)