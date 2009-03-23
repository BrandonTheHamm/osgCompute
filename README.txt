Welcome to the osgCompute library - a nodekit for OpenSceneGraph.

For up-to-date information on the project, in-depth details on how to 
compile and run libraries and examples, see the documentation on the 
osgCompute website:

    http://www.cg.informatik.uni-siegen.de/svt
  
You will find simplified build notes below. For support 
subscribe to the public mailing list of OpenSceneGraph:

    http://www.openscenegraph.org/projects/osg/wiki/MailingLists

or go directly to the OpenSceneGraph forum which is synchronized
with the mailing lists:

    http://forum.openscenegraph.org



SVT Group (Simulation and Visualization Toolkit) 
Project Lead.
19th March 2009.

--



What is osgCompute?
===============================
osgCompute is the abstract base library for the execution of code on parallel
streaming processors. The library is connected to OpenSceneGraph (OSG)
and thus it can be included in the scenegraph. It gives the user the
possibility to jump to the graphics processing unit (GPU) for any kind
of calculations. The manipulated data is then afterwards available
to the scenegraph for further processing (e.g. rendering).

osgCuda is based on the osgCompute library and implements the specific
functionality for NVIDIA's CUDA (http://www.nvidia.com/object/cuda_home.html).
CUDA is a general purpose parallel computing architecture that leverages the
parallel compute engine in NVIDIA GPUs to solve many complex computational
problems in a fraction of the time required on a CPU.

What is next? OpenCL (Open Computing Language) is a new heterogeneous
computing environment. At the moment a release of a OpenCL API/driver interface
doesn't exist. The use of osgCompute as the base library for the connection of
OpenCL to OSG looks promising. 



How to build the osgCompute
===============================

The osgCompute uses the CMake build system to generate a 
platform-specific build environment. This build system 
was choosen since OpenSceneGraph is also based on it.

If you don't already have CMake installed on your system you can grab 
it from http://www.cmake.org, use version 2.4.6 or later.  Details on the 
OpenSceneGraph's CMake build can be found at:

    http://www.openscenegraph.org/projects/osg/wiki/Build/CMake

Under unices (i.e. Linux, IRIX, Solaris, Free-BSD, HP-Ux, AIX, OSX) 
use the cmake or ccmake command-line utils. To compile osgCompute type following:

    cd osgCompute
    cmake . -DCMAKE_BUILD_TYPE=Release
    make
    sudo make install
  
Alternatively, you can create an out-of-source build directory and run 
cmake or ccmake from there. The advantage to this approach is that the 
temporary files created by CMake won't clutter the osgCompute
source directory, and also makes it possible to have multiple 
independent build targets by creating multiple build directories. In a 
directory alongside the osgCompute use:

    mkdir build
    cd build
    cmake ../ -DCMAKE_BUILD_TYPE=Release
    make
    sudo make install

If you would like to install the library somewhere else than the default
install location, so type in the main directory of osgCompute the following:

    cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install 
    make install

Under Windows use the GUI tool of the CMake setup to build your VisualStudio 
files. The following page on OpenSceneGraph's wiki dedicated to the CMake build 
system should help guide you through the process:

    http://www.openscenegraph.org/projects/osg/wiki/Support/PlatformSpecifics/VisualStudio

    
    
OpenSceneGraph-Version
===============================
OpenSceneGraph (http://www.openscenegraph.org) is currently available in the official
version: 2.8. Although osgCompute has been tested with this version it should also run with
previous versions of OpenSceneGraph.
Attention: a current SVN version of OpenSceneGraph (2.9.1 or later) is recommended
for support of multithreading with osgCuda!



CUDA-Version
===============================
osgCuda supports the Runtime API of CUDA 2.1 which also includes the debugging of the
graphics kernels. The Driver API is currently not supported by osgCuda. 



Important environment variables: 
===============================
OSGCOMPUTE_FILE_PATH
--------------------
If the examples are built please ensure that the data will be available (separate SVN checkout:
svn co https://svn.cg.informatik.uni-siegen.de/svt/osgCompute-Data/trunk osgCompute-Data).
Set the environment variable OSGCOMPUTE_FILE_PATH to the folder which contains the
osgCompute data (e.g. C:\SVT\osgCompute-Data).

You have to setup the OSG_FILE_PATH environment variable accordingly in order to find these files. 
Please add OSGCOMPUTE_FILE_PATH to OSG_FILE_PATH. This ensures that OSG is able to load the
necessary data successfully.

On Windows adapt the environment variable in the system properties and add to the OSG_FILE_PATH variable
   C:\SDK\OpenSceneGraph-Data;C:\SDK\OpenSceneGraph-Data\Images;%OSGCOMPUTE_FILE_PATH%
On Unix systems do the following:
  export OSG_FILE_PATH=$OSG_FILE_PATH:$OSGCOMPUTE_FILE_PATH


OSGCOMPUTEDIR
--------------------
The environment variable OSGCOMPUTEDIR should be defined for proper work with osgCompute.
Scripts like Findosgcompute.cmake search for this environment variable automatically (especially
useful for windows users).


CUDADIR
--------------------
The environment variable CUDADIR should be defined for proper work with CUDA.
It should point to the root path of your CUDA installation (especially useful for
windows users, e.g. C:\CUDA).
