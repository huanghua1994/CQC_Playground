# simintConfig.cmake
# ------------------
#
# SIMINT cmake module.
# This module sets the following variables in your project::
#
#   simint_FOUND - true if simint and all required components found on the system
#   simint_VERSION - simint version in format Major.Minor.Release
#   simint_INCLUDE_DIRS - Directory where simint headers are located.
#   simint_INCLUDE_DIR - same as DIRS
#   simint_LIBRARIES - simint library to link against.
#   simint_LIBRARY - same as LIBRARIES
#   simint_VECTOR - vectorization level of library.
#   simint_MAXAM - maximum angular momentum supported by the library
#
#
# Available components: shared static ::
#
#   shared - search for only shared library
#   static - search for only static library
#   am{integer} - search for library with angular momentum >= this integer
#   der{integer} - search for library with derivatives >= this integer
#
#
# Exported targets::
#
# If simint is found, this module defines the following :prop_tgt:`IMPORTED`
# target. Target is shared _or_ static, so, for both, use separate, not
# overlapping, installations. ::
#
#   simint::simint - the main simint library with header & defs attached.
#
#
# Suggested usage::
#
#   find_package(simint)
#   find_package(simint 0.7 EXACT CONFIG REQUIRED COMPONENTS shared am3 der0)
#
#
# The following variables can be set to guide the search for this package::
#
#   simint_DIR - CMake variable, set to directory containing this Config file
#   CMAKE_PREFIX_PATH - CMake variable, set to root directory of this package
#   PATH - environment variable, set to bin directory of this package
#   CMAKE_DISABLE_FIND_PACKAGE_simint - CMake variable, disables
#     find_package(simint) when not REQUIRED, perhaps to force internal build


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was simintConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(_valid_components
    shared
    static
)

set(simint_VECTOR avx)
set(simint_MAXAM 4)
set(simint_MAXDER 0)

# set the include files
set(simint_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set(simint_INCLUDE_DIRS ${simint_INCLUDE_DIR})

# set the libraries to use
set(simint_LIBRARY_BASE "${PACKAGE_PREFIX_DIR}/lib")
if(True)
    set(simint_LIBRARY_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}simint${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(simint_shared_FOUND 1)
else()
    set(simint_LIBRARY_NAME ${CMAKE_STATIC_LIBRARY_PREFIX}simint${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(simint_static_FOUND 1)
endif()

set(simint_LIBRARY "${simint_LIBRARY_BASE}/${simint_LIBRARY_NAME}")
set(simint_LIBRARIES ${simint_LIBRARY})

# Check for the max am and for invalid components
foreach(_comp IN LISTS simint_FIND_COMPONENTS)
    string(REGEX MATCH "^am[0-9]+$" _is_am_comp ${_comp})
    string(REGEX MATCH "^der[0-9]+$" _is_der_comp ${_comp})
    if(_is_am_comp)
        string(SUBSTRING ${_comp} 2 -1 _comp_reqam)

        # _comp_reqam = requested AM
        # can we satisfy that?
        if(NOT ${simint_MAXAM} LESS ${_comp_reqam})
            set(simint_${_comp}_FOUND 1)
        else()
            if(NOT CMAKE_REQUIRED_QUIET)
                message(STATUS "simintConfig missing component: requested AM ${_comp_reqam} is greater than ${simint_MAXAM} in ${simint_LIBRARY}")
            endif()
        endif()
    elseif(_is_der_comp)
        string(SUBSTRING ${_comp} 3 -1 _comp_reqder)

        # _comp_reqder = requested derivative
        # can we satisfy that?
        if(NOT ${simint_MAXDER} LESS ${_comp_reqder})
            set(simint_${_comp}_FOUND 1)
        else()
            if(NOT CMAKE_REQUIRED_QUIET)
                message(STATUS "simintConfig missing component: requested derivative ${_comp_reqder} is greater than ${simint_MAXDER} in ${simint_LIBRARY}")
            endif()
        endif()
    else()
      # Is this an otherwise valid component?
      list(FIND _valid_components ${_comp} _is_valid_comp)
      if(${_is_valid_comp} LESS 0)
          message(STATUS "simintConfig: requested invalid component: ${_comp}")
      endif()
    endif()
endforeach()


check_required_components(simint)

#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already built this as a subproject
#-----------------------------------------------------------------------------
if(NOT TARGET simint::simint)
    include("${CMAKE_CURRENT_LIST_DIR}/simintTargets.cmake")
endif()

