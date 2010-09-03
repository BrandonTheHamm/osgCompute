#
# Try to find CUDA compiler, runtime libraries, and include path.
# Once done this will define
#
# CUDA_FOUND
# CUDA_INCLUDE_PATH
# CUDA_RUNTIME_LIBRARY
# CUDA_COMPILER
#
# It will also define the following macro:
#
# WRAP_CUDA
#

# Works now also with emulation mode. Added --device-emulation to CUDA_OPTIONS. (SVT Group)


# Allow the user to specify if the device code is supposed to be 32 or 64 bit.
# copied from current nvidia texture-tools (FindCUDA.cmake)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
option(CUDA_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${CUDA_64_BIT_DEVICE_CODE_DEFAULT})


# Find cuda compiler
IF (WIN32)
	FIND_PROGRAM (CUDA_COMPILER nvcc.exe
		$ENV{CUDA_BIN_PATH}
		DOC "The CUDA Compiler")
ELSE(WIN32)
	FIND_PROGRAM (CUDA_COMPILER nvcc
		$ENV{CUDA_BIN_PATH}
		/usr/local/cuda/bin
		DOC "The CUDA Compiler")
ENDIF(WIN32)

IF (CUDA_COMPILER)
	GET_FILENAME_COMPONENT (CUDA_COMPILER_DIR ${CUDA_COMPILER} PATH)
	GET_FILENAME_COMPONENT (CUDA_COMPILER_SUPER_DIR ${CUDA_COMPILER_DIR} PATH)
ELSE (CUDA_COMPILER)
	SET (CUDA_COMPILER_DIR .)
	SET (CUDA_COMPILER_SUPER_DIR ..)
ENDIF (CUDA_COMPILER)

FIND_PATH (CUDA_INCLUDE_PATH cuda_runtime.h
	$ENV{CUDA_INC_PATH}
	${CUDA_COMPILER_SUPER_DIR}/include
	DOC "The directory where CUDA headers reside")

#cutil not needed yet - svt group
#FIND_PATH (CUTIL_INCLUDE_PATH cutil.h
#	"$ENV{NVSDKCUDA_ROOT}/common/inc"
#	"$ENV{PROGRAMFILES}/NVIDIA Corporation/NVIDIA CUDA SDK/common/inc"
#	DOC "The directory where the CUTIL headers reside")

FIND_LIBRARY (CUDA_RUNTIME_LIBRARY
	NAMES cudart
	PATHS
	$ENV{CUDA_LIB_PATH}
	${CUDA_COMPILER_SUPER_DIR}/lib
	${CUDA_COMPILER_DIR}
	DOC "The CUDA runtime library")


#cutil not needed yet - svt group
# CUDA_CUT_LIBRARIES

# cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# to get these confused, so we are setting the name based on the word size of
# the build.
#if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#  set(cuda_cutil_name cutil64)
#else(CMAKE_SIZEOF_VOID_P EQUAL 8)
#  set(cuda_cutil_name cutil32)
#endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

#FIND_LIBRARY (CUTIL_LIBRARY
#	NAMES ${cuda_cutil_name}
#	PATHS
#	"$ENV{NVSDKCUDA_ROOT}/common/lib"
#	"$ENV{PROGRAMFILES}/NVIDIA Corporation/NVIDIA CUDA SDK/common/lib"
#	DOC "The CUTIL library")

IF (CUDA_INCLUDE_PATH AND CUDA_RUNTIME_LIBRARY)
	SET (CUDA_FOUND TRUE)
ELSE (CUDA_INCLUDE_PATH AND CUDA_RUNTIME_LIBRARY)
	SET (CUDA_FOUND FALSE)
ENDIF (CUDA_INCLUDE_PATH AND CUDA_RUNTIME_LIBRARY)

SET (CUDA_LIBRARIES ${CUDA_RUNTIME_LIBRARY})

MARK_AS_ADVANCED (CUDA_FOUND CUDA_COMPILER CUDA_RUNTIME_LIBRARY)

#cutil not needed yet - svt group
#IF (CUTIL_INCLUDE_PATH AND CUTIL_LIBRARY)
#	SET (CUTIL_FOUND 1 CACHE STRING "Set to 1 if CUDA is found, 0 otherwise")
#ELSE (CUTIL_INCLUDE_PATH AND CUTIL_LIBRARY)
#	SET (CUTIL_FOUND 0 CACHE STRING "Set to 1 if CUDA is found, 0 otherwise")
#ENDIF (CUTIL_INCLUDE_PATH AND CUTIL_LIBRARY)

#SET (CUTIL_LIBRARIES ${CUTIL_LIBRARY})

#MARK_AS_ADVANCED (CUTIL_FOUND)


# copied from current nvidia texture-tools (FindCUDA.cmake)
if(CUDA_64_BIT_DEVICE_CODE)
    set(CUDA_OPTIONS -m64)
else()
    set(CUDA_OPTIONS -m32)
endif()

# in Cuda 3.0 not needed anymore - svt group
# You may use this option for proper debugging in emulation mode.
#option(CUDA_HOST_COMPILATION_C "Generated file extension. You may use this option for proper debugging in emulation mode." OFF)
#IF (CUDA_HOST_COMPILATION_C)
#	SET(CUDA_OPTIONS ${CUDA_OPTIONS} --host-compilation=C)
#ENDIF (CUDA_HOST_COMPILATION_C)


# we use a separate application / exmaple for emulation - therefore: not needed anymore - svt group
#OPTION(CUDA_EMULATION "Use CUDA emulation mode. Attention: this enables debugging of CUDA kernels on the CPU." OFF)
#IF (CUDA_EMULATION)
#	SET (CUDA_OPTIONS ${CUDA_OPTIONS} --device-emulation --define-macro=_DEVICEEMU --debug)
#ENDIF (CUDA_EMULATION)

# in Cuda 3.0 not needed anymore - svt group
#OPTION(CUDA_USE_GEN_C_FILE_EXTENSION "Generated files will have the extension .gen.c instead of .gen.cpp" OFF)


#specify additional/system specific CUDA options by user
SET (CUDA_NVCC_USER_OPTIONS
    "-arch sm_10"
	CACHE
	STRING
	"Set additional user specific compiler options to Cuda nvcc" 
)

# put CUDA_NVCC_USER_OPTIONS into a cmake list for further processing (otherwise problems with quotation marks)
#SEPARATE_ARGUMENTS(CUDA_NVCC_USER_OPTIONS_CONVERTED WINDOWS_COMMAND ${CUDA_NVCC_USER_OPTIONS}) # works only since cmake 2.8
# therefore use the follwing line
# Convert the value of CUDA_NVCC_USER_OPTIONS to a semi-colon separated list. All spaces are replaced with ';'.
SEPARATE_ARGUMENTS(CUDA_NVCC_USER_OPTIONS)

#copy user options to nsight options
SET (CUDA_OPTIONS ${CUDA_OPTIONS} ${CUDA_NVCC_USER_OPTIONS} )

IF(WIN32)
    SET (CUDA_OPTIONS ${CUDA_OPTIONS} --define-macro=WIN32)
ENDIF(WIN32)


# Get include directories.
MACRO(GET_CUDA_INC_DIRS _cuda_INC_DIRS)
	SET(${_cuda_INC_DIRS})
	GET_DIRECTORY_PROPERTY(_inc_DIRS INCLUDE_DIRECTORIES)

	FOREACH(_current ${_inc_DIRS})
		SET(${_cuda_INC_DIRS} ${${_cuda_INC_DIRS}} "-I" ${_current})
	ENDFOREACH(_current ${_inc_DIRS})
	
	SET(${_cuda_INC_DIRS} ${${_cuda_INC_DIRS}} "-I" ${CUDA_INCLUDE_PATH})

#	IF (CMAKE_SYTEM_INCLUDE_PATH)
#		SET(${_cuda_INC_DIRS} ${${_cuda_INC_DIRS}} "-I" ${CMAKE_SYSTEM_INCLUDE_PATH})
#	ENDIF (CMAKE_SYTEM_INCLUDE_PATH)
#	IF (CMAKE_INCLUDE_PATH)
#		SET(${_cuda_INC_DIRS} ${${_cuda_INC_DIRS}} "-I" ${CMAKE_INCLUDE_PATH})
#	ENDIF (CMAKE_INCLUDE_PATH)

ENDMACRO(GET_CUDA_INC_DIRS)


# Get file dependencies.
MACRO (GET_CUFILE_DEPENDENCIES dependencies file)
	GET_FILENAME_COMPONENT(filepath ${file} PATH)
	
	#  parse file for dependencies
	FILE(READ "${file}" CONTENTS)
	#STRING(REGEX MATCHALL "#[ \t]*include[ \t]+[<\"][^>\"]*" DEPS "${CONTENTS}")
	STRING(REGEX MATCHALL "#[ \t]*include[ \t]+\"[^\"]*" DEPS "${CONTENTS}")
	
	SET(${dependencies})
	
	FOREACH(DEP ${DEPS})
		STRING(REGEX REPLACE "#[ \t]*include[ \t]+\"" "" DEP "${DEP}")

		FIND_PATH(PATH_OF_${DEP} ${DEP}
			${filepath})

		IF(NOT ${PATH_OF_${DEP}} STREQUAL PATH_OF_${DEP}-NOTFOUND)
			#MESSAGE("${file} : ${PATH_OF_${DEP}}/${DEP}")
			SET(${dependencies} ${${dependencies}} ${PATH_OF_${DEP}}/${DEP})
		ENDIF(NOT ${PATH_OF_${DEP}} STREQUAL PATH_OF_${DEP}-NOTFOUND)
		
	ENDFOREACH(DEP)
    
    # Set additional dependencies, therefore read CUDA_PROJECT_DEPENDENCIES
    #MESSAGE("${CUDA_PROJECT_DEPENDENCIES}")
	FOREACH(DEP ${CUDA_PROJECT_DEPENDENCIES})
		SET(${dependencies} ${${dependencies}} ${DEP})
		#MESSAGE("${DEP} added")
	ENDFOREACH(DEP)

ENDMACRO (GET_CUFILE_DEPENDENCIES)


# WRAP_CUDA(outfile ...)
MACRO (WRAP_CUDA outfiles)
	GET_CUDA_INC_DIRS(cuda_includes)
	#MESSAGE(${cuda_includes})

    # in Cuda 3.0 not needed anymore - svt group
    #check for c-file extension
    # this may be important when using emulation mode with CUDA_HOST_COMPILATION_C
    #if(CUDA_USE_GEN_C_FILE_EXTENSION)
    #    set(CUDA_GEN_FILE_EXTENSION c)
    #else()
    #    set(CUDA_GEN_FILE_EXTENSION cpp)
    #endif()
    
	FOREACH (CUFILE ${ARGN})
		GET_FILENAME_COMPONENT (CUFILE ${CUFILE} ABSOLUTE)
		GET_FILENAME_COMPONENT (CPPFILE ${CUFILE} NAME_WE)
		SET (CPPFILE ${CMAKE_CURRENT_BINARY_DIR}/${CPPFILE}.gen.cpp)
        #SET (CPPFILE ${CMAKE_CURRENT_BINARY_DIR}/${CPPFILE}.gen.${CUDA_GEN_FILE_EXTENSION})

		GET_CUFILE_DEPENDENCIES(CUDEPS ${CUFILE})
		#MESSAGE("${CUDEPS}")

		ADD_CUSTOM_COMMAND (
			OUTPUT ${CPPFILE}
			COMMAND ${CUDA_COMPILER}
			ARGS -cuda ${cuda_includes} ${CUDA_OPTIONS} -o ${CPPFILE} ${CUFILE}
			MAIN_DEPENDENCY ${CUFILE}
			DEPENDS ${CUDEPS})

		#MACRO_ADD_FILE_DEPENDENCIES(${CUFILE} ${CPPFILE})

		SET (${outfiles} ${${outfiles}} ${CPPFILE})
	ENDFOREACH (CUFILE)
	
	SET_SOURCE_FILES_PROPERTIES(${outfiles} PROPERTIES GENERATED 1)
	
ENDMACRO (WRAP_CUDA)