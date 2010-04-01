#
# Try to find CUDA compiler, runtime libraries, and include path.
# Once done this will define
#
# CUDA_EMU_FOUND
# CUDA_EMU_INCLUDE_PATH
# CUDA_EMU_RUNTIME_LIBRARY
# CUDA_EMU_COMPILER
#
# It will also define the following macro:
#
# WRAP_CUDA_EMU
#

# Works now also with emulation mode. Added --device-emulation to CUDA_OPTIONS. (SVT Group)


# Allow the user to specify if the device code is supposed to be 32 or 64 bit.
# copied from current nvidia texture-tools (FindCUDA.cmake)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(CUDA_EMU_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(CUDA_EMU_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
option(CUDA_EMU_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${CUDA_EMU_64_BIT_DEVICE_CODE_DEFAULT})


# Find cuda compiler
IF (WIN32)
	FIND_PROGRAM (CUDA_EMU_COMPILER nvcc.exe
		$ENV{CUDA_BIN_PATH}
		DOC "The CUDA Emulation Compiler")
ELSE(WIN32)
	FIND_PROGRAM (CUDA_EMU_COMPILER nvcc
		$ENV{CUDA_BIN_PATH}
		/usr/local/cuda/bin
		DOC "The CUDA Compiler")
ENDIF(WIN32)

IF (CUDA_EMU_COMPILER)
	GET_FILENAME_COMPONENT (CUDA_EMU_COMPILER_DIR ${CUDA_EMU_COMPILER} PATH)
	GET_FILENAME_COMPONENT (CUDA_EMU_COMPILER_SUPER_DIR ${CUDA_EMU_COMPILER_DIR} PATH)
ELSE (CUDA_EMU_COMPILER)
	SET (CUDA_EMU_COMPILER_DIR .)
	SET (CUDA_EMU_COMPILER_SUPER_DIR ..)
ENDIF (CUDA_EMU_COMPILER)

FIND_PATH (CUDA_EMU_INCLUDE_PATH cuda_runtime.h
	$ENV{CUDA_INC_PATH}
	${CUDA_EMU_COMPILER_SUPER_DIR}/include
	DOC "The directory where CUDA headers reside")

FIND_LIBRARY (CUDA_EMU_RUNTIME_LIBRARY
	NAMES cudartemu
	PATHS
	$ENV{CUDA_LIB_PATH}
	${CUDA_EMU_COMPILER_SUPER_DIR}/lib
	${CUDA_EMU_COMPILER_DIR}
	DOC "The CUDA runtime library")



IF (CUDA_EMU_INCLUDE_PATH AND CUDA_EMU_RUNTIME_LIBRARY)
	SET (CUDA_EMU_FOUND TRUE)
ELSE (CUDA_EMU_INCLUDE_PATH AND CUDA_EMU_RUNTIME_LIBRARY)
	SET (CUDA_EMU_FOUND FALSE)
ENDIF (CUDA_EMU_INCLUDE_PATH AND CUDA_EMU_RUNTIME_LIBRARY)

SET (CUDA_EMU_LIBRARIES ${CUDA_EMU_RUNTIME_LIBRARY})

MARK_AS_ADVANCED (CUDA_EMU_FOUND CUDA_EMU_COMPILER CUDA_EMU_RUNTIME_LIBRARY)



# copied from current nvidia texture-tools (FindCUDA.cmake)
if(CUDA_EMU_64_BIT_DEVICE_CODE)
    set(CUDA_EMU_OPTIONS -m64)
else()
    set(CUDA_EMU_OPTIONS -m32)
endif()

# we are here in emulation mode! Thus, use emulation comiler options
SET (CUDA_EMU_OPTIONS ${CUDA_EMU_OPTIONS} --device-emulation --define-macro=_DEVICEEMU --debug)

IF(WIN32)
    SET (CUDA_EMU_OPTIONS ${CUDA_EMU_OPTIONS} --define-macro=WIN32)
ENDIF(WIN32)


# Get include directories.
MACRO(GET_CUDA_EMU_INC_DIRS _cuda_emu_INC_DIRS)
	SET(${_cuda_emu_INC_DIRS})
	GET_DIRECTORY_PROPERTY(_inc_DIRS INCLUDE_DIRECTORIES)

	FOREACH(_current ${_inc_DIRS})
		SET(${_cuda_emu_INC_DIRS} ${${_cuda_emu_INC_DIRS}} "-I" ${_current})
	ENDFOREACH(_current ${_inc_DIRS})
	
	SET(${_cuda_emu_INC_DIRS} ${${_cuda_emu_INC_DIRS}} "-I" ${CUDA_EMU_INCLUDE_PATH})

ENDMACRO(GET_CUDA_EMU_INC_DIRS)


# Get file dependencies.
MACRO (GET_CUEMUFILE_DEPENDENCIES dependencies file)
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

ENDMACRO (GET_CUEMUFILE_DEPENDENCIES)


# WRAP_CUDA_EMU(outfile ...)
MACRO (WRAP_CUDA_EMU outfiles)
	GET_CUDA_EMU_INC_DIRS(cuda_emu_includes)
	#MESSAGE(${cuda_emu_includes})

	FOREACH (CUEMUFILE ${ARGN})
		GET_FILENAME_COMPONENT (CUEMUFILE ${CUEMUFILE} ABSOLUTE)
		GET_FILENAME_COMPONENT (CPPFILE ${CUEMUFILE} NAME_WE)
		SET (CPPFILE ${CMAKE_CURRENT_BINARY_DIR}/${CPPFILE}.gen.cpp)


		GET_CUEMUFILE_DEPENDENCIES(CUEMUDEPS ${CUEMUFILE})
		#MESSAGE("${CUEMUDEPS}")

		ADD_CUSTOM_COMMAND (
			OUTPUT ${CPPFILE}
			COMMAND ${CUDA_EMU_COMPILER}
			ARGS -cuda ${cuda_emu_includes} ${CUDA_EMU_OPTIONS} -o ${CPPFILE} ${CUEMUFILE}
			MAIN_DEPENDENCY ${CUEMUFILE}
			DEPENDS ${CUEMUDEPS})

		#MACRO_ADD_FILE_DEPENDENCIES(${CUEMUFILE} ${CPPFILE})

		SET (${outfiles} ${${outfiles}} ${CPPFILE})
	ENDFOREACH (CUEMUFILE)
	
	SET_SOURCE_FILES_PROPERTIES(${outfiles} PROPERTIES GENERATED 1)
	
ENDMACRO (WRAP_CUDA_EMU)