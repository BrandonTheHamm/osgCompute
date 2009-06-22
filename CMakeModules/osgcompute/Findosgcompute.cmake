# This file adapts the Findosg* files to the specific needs of finding osgCompute-modules.
# -SVT Group
# 
# Locate osgCompute
# This module defines
# OSGCOMPUTE_LIBRARY and OSGCOMPUTE_LIBRARY_DEBUG
# OSGCOMPUTE_FOUND, if false, do not try to link to osgCompute
# OSGCOMPUTE_INCLUDE_DIR, where to find the headers
#
# $OSGCOMPUTEDIR is an environment variable that points to the
# root directory of the osgCompute distribution.
#
# Created by Eric Wing for osg. Adapted by SVT Group for osgCompute.

# Header files are presumed to be included like
#include <osgCompute/Param>

include(Findosgcompute_functions)
OSGCOMPUTE_FIND_PATH   (OSGCOMPUTE osgCompute/Context)
OSGCOMPUTE_FIND_LIBRARY(OSGCOMPUTE osgCompute)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSGCOMPUTE DEFAULT_MSG OSGCOMPUTE_LIBRARY OSGCOMPUTE_INCLUDE_DIR)
