#######################################################
# prepare Examples
#######################################################
SET(TARGET_DEFAULT_PREFIX "example_")
SET(TARGET_DEFAULT_LABEL_PREFIX "Examples")

#set data path to env
file(TO_CMAKE_PATH "$ENV{OSGCOMPUTE_FILE_PATH}" DATA_PATH)


###############################
# set libs which are commonly useful
###############################
SET(TARGET_COMMON_LIBRARIES 
)


# cuda, osg needed
##################################
IF ( CUDA_FOUND AND OSG_FOUND )
  ADD_SUBDIRECTORY(osgEndiannessDemo)
  ADD_SUBDIRECTORY(osgGeometryDemo)
  ADD_SUBDIRECTORY(osgParticleDemo)
  ADD_SUBDIRECTORY(osgTexDemo)
  ADD_SUBDIRECTORY(osgRTTDemo)
  ADD_SUBDIRECTORY(osgTraceDemo)
ENDIF( CUDA_FOUND AND OSG_FOUND )