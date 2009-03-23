#set data path to env
file(TO_CMAKE_PATH "$ENV{OSGCOMPUTE_FILE_PATH}" DATA_PATH)

# setup the base module
ADD_SUBDIRECTORY(osgCompute)

# if cuda is available the cuda module will be setup
IF (CUDA_FOUND)
  ADD_SUBDIRECTORY(osgCuda)
ENDIF(CUDA_FOUND)