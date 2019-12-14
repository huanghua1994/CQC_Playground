#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "simint::simint" for configuration "Release"
set_property(TARGET simint::simint APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(simint::simint PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsimint.so"
  IMPORTED_SONAME_RELEASE "libsimint.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS simint::simint )
list(APPEND _IMPORT_CHECK_FILES_FOR_simint::simint "${_IMPORT_PREFIX}/lib/libsimint.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
