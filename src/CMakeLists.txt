#set data path to env
file(TO_CMAKE_PATH "$ENV{OSGCOMPUTE_FILE_PATH}" DATA_PATH)

# setup the base module
ADD_SUBDIRECTORY(osgCompute)

# if cuda is available the cuda module will be setup,
# and if cuda emulation is available then also osgCudaEmu
IF (CUDA_FOUND)
  ADD_SUBDIRECTORY(osgCuda)
  #IF ( CUDA_EMU_FOUND )
  #  ADD_SUBDIRECTORY(osgCudaEmu)
  #ENDIF ( CUDA_EMU_FOUND )
ENDIF(CUDA_FOUND)

IF( CUDA_NSIGHT_FOUND AND BUILD_CUDA_NSIGHT_SUPPORT )
    ADD_SUBDIRECTORY(osgCudaNsight)
ENDIF( CUDA_NSIGHT_FOUND AND BUILD_CUDA_NSIGHT_SUPPORT )