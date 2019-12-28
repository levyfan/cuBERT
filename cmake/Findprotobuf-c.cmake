include(FindPackageHandleStandardArgs)

# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
set( _protobuf_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES .a )

find_path(protobuf-c_INCLUDE_DIRS NAMES protobuf-c/protobuf-c.h)
find_library(protobuf-c_STATIC_LIBRARIES NAMES protobuf-c)
find_program(protobuf-c_PROTOC_EXECUTABLE NAMES protoc-c)

find_package_handle_standard_args(protobuf-c DEFAULT_MSG
        protobuf-c_STATIC_LIBRARIES
        protobuf-c_INCLUDE_DIRS
        protobuf-c_PROTOC_EXECUTABLE)

mark_as_advanced(
        protobuf-c_INCLUDE_DIRS
        protobuf-c_STATIC_LIBRARIES
        protobuf-c_PROTOC_EXECUTABLE)

# Restore the original find library ordering
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_protobuf_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
