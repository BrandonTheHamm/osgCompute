# This file adapts the Findosg* files to the specific needs of finding osgCompute-modules.
# Here: the osgCuda module!
# -SVT Group
# 
# Locate osgCuda
# This module defines
# OSGCUDA_LIBRARY and OSGCUDA_LIBRARY_DEBUG
# OSGCUDA_FOUND, if false, do not try to link to osgCuda
# OSGCUDA_INCLUDE_DIR, where to find the headers
#
# $OSGCOMPUTEDIR is an environment variable that points to the
# root directory of the osgCompute distribution.
#
# Created by Eric Wing for osg. Adapted by SVT Group for osgCompute.

# Header files are presumed to be included like
#include <osgCuda/Contex>

include(Findosgcompute_functions)
OSGCOMPUTE_FIND_PATH   (OSGCUDA osgCuda/Context)
OSGCOMPUTE_FIND_LIBRARY(OSGCUDA osgCuda)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSGCUDA DEFAULT_MSG OSGCUDA_LIBRARY OSGCUDA_INCLUDE_DIR)
