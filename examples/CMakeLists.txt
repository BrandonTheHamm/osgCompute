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
  ADD_SUBDIRECTORY(osgTexDemo)
  ADD_SUBDIRECTORY(osgParticleDemo)
  ADD_SUBDIRECTORY(osgEndiannessDemo)
  ADD_SUBDIRECTORY(osgGeometryDemo)
  IF( CUDA_EMU_FOUND AND BUILD_EMU_EXAMPLES )
    ADD_SUBDIRECTORY(osgGeometryEmu)
  ENDIF( CUDA_EMU_FOUND AND BUILD_EMU_EXAMPLES )
ENDIF( CUDA_FOUND AND OSG_FOUND )