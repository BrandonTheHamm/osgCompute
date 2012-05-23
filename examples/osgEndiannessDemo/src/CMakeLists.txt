#########################################################################
# Set target name und set path to data folder of the target
#########################################################################

SET(TARGETNAME osgEndiannessDemo)
SET(TARGET_DATA_PATH "${DATA_PATH}/${TARGETNAME}")


#########################################################################
# Do necessary checking stuff (check for other libraries to link against ...)
#########################################################################

# find osg
INCLUDE(Findosg)
INCLUDE(FindosgUtil)
INCLUDE(FindOpenThreads)
# check for cuda
INCLUDE(FindCuda)

# if needed then specify computing model, e.g.:
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_11)

#Uncomment to enable CUDA Debugging via Parallel NSight
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G)


#########################################################################
# Set basic include directories
#########################################################################

# set include dirs
SET(HEADER_PATH ${osgCompute_SOURCE_DIR}/examples/${TARGETNAME}/include)
INCLUDE_DIRECTORIES(
    ${HEADER_PATH}
    ${OSG_INCLUDE_DIR}
    ${CUDA_TOOLKIT_INCLUDE}
)


#########################################################################
# Collect header and source files and process macros
#########################################################################

# collect all headers

SET(TARGET_H
)


SET(MY_CUDA_SOURCE_FILES
	main.cu
)

# Use the CUDA_COMPILE macro.
CUDA_COMPILE( CUDA_FILES ${MY_CUDA_SOURCE_FILES} )

# collect the sources
SET(TARGET_SRC
	main.cpp
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

# Setup groups for resources 

# First: collect the necessary files which were not collected up to now
# Therefore, fill the following variables: 
# MY_ICE_FILES - MY_MODEL_FILES - MY_SHADER_FILES - MY_UI_FILES - MY_XML_FILES

# collect shader files
#SET(MY_SHADER_FILES
#)

# finally, use module to build groups
INCLUDE(GroupInstall)


# now set up the ADDITIONAL_FILES variable to ensure that the files will be visible in the project
# and/or that they are forwarded to the linking stage
SET(ADDITIONAL_FILES
	#${MY_SHADER_FILES}
	${CUDA_FILES}
)



#########################################################################
# Setup libraries to link against
#########################################################################

# put here own project libraries, for example. (Attention: you do not have
# to differentiate between debug and optimized: this is done automatically by cmake
SET(TARGET_ADDITIONAL_LIBRARIES
	osgCompute
	osgCuda
	osgCudaInit
)


# put here the libraries which are collected in a variable (i.e. most of the FindXXX scrips)
# the macro (LINK_WITH_VARIABLES) ensures that also the ${varname}_DEBUG names will resolved correctly
SET(TARGET_VARS_LIBRARIES 	
	OPENTHREADS_LIBRARY
	OSG_LIBRARY
	OSGUTIL_LIBRARY
    CUDA_CUDART_LIBRARY
)


#########################################################################
# Example setup and install
#########################################################################

# this is a user definded macro which does all the work for us
# it also takes into account the variables TARGET_SRC,
# TARGET_H and TARGET_ADDITIONAL_LIBRARIES and TARGET_VARS_LIBRARIES and ADDITIONAL_FILES
SETUP_EXAMPLE(${TARGETNAME})
