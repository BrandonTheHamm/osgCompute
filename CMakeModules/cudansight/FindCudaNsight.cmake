#
# Try to find Nvidia NSIGHT (aka Nexus) compiler, runtime libraries, and include path.
# Once done this will define
#
# CUDA_NSIGHT_FOUND
# CUDA_NSIGHT_INCLUDE_PATH
# CUDA_NSIGHT_RUNTIME_LIBRARY
# CUDA_NSIGHT_COMPILER
#
# It will also define the following macro:
#
# WRAP_CUDA_NSIGHT
#

# Now in alpha state for NSIGHT testing (beta 1 toolkit SDK) (SVT Group)

# LOOKING FOR ENV VAR: CUDANSIGHTDIR - which points at the moment to the CUDA dir of NSIGHT!

# Allow the user to specify if the device code is supposed to be 32 or 64 bit.
# copied from current nvidia texture-tools (FindCUDA.cmake)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(CUDA_NSIGHT_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(CUDA_NSIGHT_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
option(CUDA_NSIGHT_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${CUDA_NSIGHT_64_BIT_DEVICE_CODE_DEFAULT})


# Find cuda compiler
IF (WIN32)
	FIND_PROGRAM (CUDA_NSIGHT_COMPILER nvcc.exe
		HINTS
		   ENV CUDANSIGHTDIR
		PATH_SUFFIXES bin
		PATHS
		   ENV CUDANSIGHTDIR
		   [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;CUDANSIGHTDIR]
		DOC "The CUDA Debugging Compiler")
ELSE(WIN32)
	FIND_PROGRAM (CUDA_NSIGHT_COMPILER nvcc
		$ENV{CUDANSIGHTDIR}/bin
		/usr/local/nsight/cuda/bin
		DOC "The CUDA Compiler")
ENDIF(WIN32)

IF (CUDA_NSIGHT_COMPILER)
	GET_FILENAME_COMPONENT (CUDA_NSIGHT_COMPILER_DIR ${CUDA_NSIGHT_COMPILER} PATH)
	GET_FILENAME_COMPONENT (CUDA_NSIGHT_COMPILER_SUPER_DIR ${CUDA_NSIGHT_COMPILER_DIR} PATH)
ELSE (CUDA_NSIGHT_COMPILER)
	SET (CUDA_NSIGHT_COMPILER_DIR .)
	SET (CUDA_NSIGHT_COMPILER_SUPER_DIR ..)
ENDIF (CUDA_NSIGHT_COMPILER)

FIND_PATH (CUDA_NSIGHT_INCLUDE_PATH cuda_runtime.h
	$ENV{CUDANSIGHTDIR}/include
	${CUDA_NSIGHT_COMPILER_SUPER_DIR}/include
	DOC "The directory where CUDA Nsight headers reside")

FIND_LIBRARY (CUDA_NSIGHT_RUNTIME_LIBRARY
	NAMES cudart
	PATHS
	$ENV{CUDANSIGHTDIR}/lib
	${CUDA_NSIGHT_COMPILER_SUPER_DIR}/lib
	${CUDA_NSIGHT_COMPILER_DIR}
	DOC "The CUDA Nsight runtime library")



IF (CUDA_NSIGHT_INCLUDE_PATH AND CUDA_NSIGHT_RUNTIME_LIBRARY)
	SET (CUDA_NSIGHT_FOUND TRUE)
ELSE (CUDA_NSIGHT_INCLUDE_PATH AND CUDA_NSIGHT_RUNTIME_LIBRARY)
	SET (CUDA_NSIGHT_FOUND FALSE)
ENDIF (CUDA_NSIGHT_INCLUDE_PATH AND CUDA_NSIGHT_RUNTIME_LIBRARY)

SET (CUDA_NSIGHT_LIBRARIES ${CUDA_NSIGHT_RUNTIME_LIBRARY})

MARK_AS_ADVANCED (CUDA_NSIGHT_FOUND CUDA_NSIGHT_COMPILER CUDA_NSIGHT_RUNTIME_LIBRARY)



# copied from current nvidia texture-tools (FindCUDA.cmake)
if(CUDA_NSIGHT_64_BIT_DEVICE_CODE)
    set(CUDA_NSIGHT_OPTIONS -m64)
else()
    set(CUDA_NSIGHT_OPTIONS -m32)
endif()

# we are here in debugging mode! Thus, use debugging comiler options.
# CAUTION: At the moment just testing with fixed options like in the nsight sample programs!
SET (CUDA_NSIGHT_OPTIONS ${CUDA_NSIGHT_OPTIONS} -G0 -D_NSIGHT_DEBUG -g )


#specify additional/system specific NSIGHT options by user
SET (CUDA_NSIGHT_NVCC_USER_OPTIONS
	"-arch sm_11 -maxrregcount=32"
	CACHE
	STRING
	"Set additional user specific compiler options to Cuda Nsight nvcc" 
)

# put CUDA_NSIGHT_NVCC_USER_OPTIONS into a cmake list for further processing (otherwise problems with quotation marks)
SEPARATE_ARGUMENTS(CUDA_NSIGHT_NVCC_USER_OPTIONS_CONVERTED WINDOWS_COMMAND ${CUDA_NSIGHT_NVCC_USER_OPTIONS})

#copy user options to nsight options
SET (CUDA_NSIGHT_OPTIONS ${CUDA_NSIGHT_OPTIONS} ${CUDA_NSIGHT_NVCC_USER_OPTIONS_CONVERTED} )

# maik - check this: not needed anymore?
IF(WIN32)
    SET (CUDA_NSIGHT_OPTIONS ${CUDA_NSIGHT_OPTIONS} --define-macro=WIN32)
ENDIF(WIN32)


# Get include directories.
MACRO(GET_CUDA_NSIGHT_INC_DIRS _cuda_nsight_INC_DIRS)
	SET(${_cuda_nsight_INC_DIRS})
	GET_DIRECTORY_PROPERTY(_inc_DIRS INCLUDE_DIRECTORIES)

	FOREACH(_current ${_inc_DIRS})
		SET(${_cuda_nsight_INC_DIRS} ${${_cuda_nsight_INC_DIRS}} "-I" ${_current})
	ENDFOREACH(_current ${_inc_DIRS})
	
	SET(${_cuda_nsight_INC_DIRS} ${${_cuda_nsight_INC_DIRS}} "-I" ${CUDA_NSIGHT_INCLUDE_PATH})

ENDMACRO(GET_CUDA_NSIGHT_INC_DIRS)


# Get file dependencies.
MACRO (GET_CUNSIGHTFILE_DEPENDENCIES dependencies file)
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

ENDMACRO (GET_CUNSIGHTFILE_DEPENDENCIES)


# WRAP_CUDA_NSIGHT(outfile ...)
MACRO (WRAP_CUDA_NSIGHT outfiles)
	GET_CUDA_NSIGHT_INC_DIRS(cuda_nsight_includes)
	#MESSAGE(${cuda_nsight_includes})

	FOREACH (CUNSIGHTFILE ${ARGN})
		GET_FILENAME_COMPONENT (CUNSIGHTFILE ${CUNSIGHTFILE} ABSOLUTE)
		GET_FILENAME_COMPONENT (CPPFILE ${CUNSIGHTFILE} NAME_WE)
		SET (CPPFILE ${CMAKE_CURRENT_BINARY_DIR}/${CPPFILE}.gen.cpp)


		GET_CUNSIGHTFILE_DEPENDENCIES(CUNSIGHTDEPS ${CUNSIGHTFILE})
		#MESSAGE("${CUNSIGHTDEPS}")

		ADD_CUSTOM_COMMAND (
			OUTPUT ${CPPFILE}
			COMMAND ${CUDA_NSIGHT_COMPILER}
			ARGS -cuda ${cuda_nsight_includes} ${CUDA_NSIGHT_OPTIONS} -o ${CPPFILE} ${CUNSIGHTFILE}
			MAIN_DEPENDENCY ${CUNSIGHTFILE}
			DEPENDS ${CUNSIGHTDEPS})

		#MACRO_ADD_FILE_DEPENDENCIES(${CUNSIGHTFILE} ${CPPFILE})

		SET (${outfiles} ${${outfiles}} ${CPPFILE})
	ENDFOREACH (CUNSIGHTFILE)
	
	SET_SOURCE_FILES_PROPERTIES(${outfiles} PROPERTIES GENERATED 1)
	
ENDMACRO (WRAP_CUDA_NSIGHT)