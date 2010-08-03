#########################################################################
# Set library and plugin name
#########################################################################

SET(LIB_NAME osgdb_serializers_osgCuda)
SET(PLUGIN_NAME osgCudaSerializer)
SET(LIB_NAME_LABEL "Plugins ${PLUGIN_NAME}")


#########################################################################
# Do necessary checking stuff
#########################################################################

INCLUDE(FindOpenThreads)
INCLUDE(Findosg)
INCLUDE(FindosgDB)


#########################################################################
# Set basic include directories
#########################################################################

INCLUDE_DIRECTORIES(
	${OSG_INCLUDE_DIR}
    ${OSGCOMPUTE_INCLUDE_DIR}
    ${OSGCUDA_INCLUDE_DIR}
    ${CUDA_INCLUDE_PATH}
)

#########################################################################
# Collect header and source files
#########################################################################

# collect all headers

SET(TARGET_H
)


# collect the sources
SET(TARGET_SRC
	Computation.cpp
	Buffer.cpp
	Array.cpp
	Memory.cpp
	Resource.cpp
	Geometry.cpp
	Texture2D.cpp
	Texture3D.cpp
	TextureRectangle.cpp
)


#########################################################################
# Setup groups for resources (mainly for MSVC project folders)
#########################################################################

# probably not needed here in a plugin lib
# for more detailed information and sample script/code see
# another library which is not a plugin


#########################################################################
# Build Library and prepare install scripts
#########################################################################

IF(DYNAMIC_LINKING)
    ADD_LIBRARY(${LIB_NAME} MODULE ${TARGET_SRC} ${TARGET_H})
ELSE (DYNAMIC_LINKING)
    ADD_LIBRARY(${LIB_NAME} STATIC ${TARGET_SRC} ${TARGET_H})
ENDIF(DYNAMIC_LINKING)


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
	OSGUTIL_LIBRARY
	OSGDB_LIBRARY
    CUDA_RUNTIME_LIBRARY
)


# Do not use 'lib' prefix in e.g. linux - this
# is a common rule for all plugins in OSG
SET_TARGET_PROPERTIES( ${LIB_NAME} PROPERTIES PREFIX "" )


INCLUDE(ModuleInstall OPTIONAL)