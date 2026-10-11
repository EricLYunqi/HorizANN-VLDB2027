cmake_minimum_required(VERSION 3.16)
project(DistributedHNSW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(PkgConfig REQUIRED)

# OpenMP
# Apple Clang supports OpenMP but Apple doesn't ship the runtime; point CMake
# at Homebrew's keg-only libomp so find_package can locate it. No-op on Linux.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(HOMEBREW_LIBOMP_PREFIX AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
    set(OpenMP_ROOT "${HOMEBREW_LIBOMP_PREFIX}")
  else()
    message(FATAL_ERROR "libomp not found. Install it with: brew install libomp")
  endif()
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found")
endif()

include(FetchContent)

# gRPC and Protobuf
option(USE_SYSTEM_GRPC "Use system installed gRPC" OFF)
if(USE_SYSTEM_GRPC)
  find_package(Protobuf REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  set(PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
  if(NOT GRPC_CPP_PLUGIN)
    message(FATAL_ERROR "grpc_cpp_plugin not found. Please install gRPC.")
  endif()
else()
  FetchContent_Declare(
    gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG v1.74.0
  )
  set(FETCHCONTENT_QUIET OFF)
  set(ABSL_ENABLE_INSTALL ON)
  FetchContent_MakeAvailable(gRPC)

  add_library(gRPC::grpc++ ALIAS grpc++)
  add_library(gRPC::grpc++_reflection ALIAS grpc++_reflection)

  set(PROTOC_TARGET "")
  if(TARGET protobuf::protoc)
    set(PROTOC_TARGET protobuf::protoc)
  elseif(TARGET protoc)
    set(PROTOC_TARGET protoc)
  endif()

  if(NOT PROTOC_TARGET)
    find_program(PROTOC_EXECUTABLE
      NAMES protoc
      PATHS ${CMAKE_BINARY_DIR}/_deps/protobuf-build
            ${CMAKE_BINARY_DIR}/_deps/grpc-build/third_party/protobuf
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT PROTOC_TARGET AND NOT PROTOC_EXECUTABLE)
    find_program(PROTOC_EXECUTABLE protoc)
    if(PROTOC_EXECUTABLE)
      message(WARNING "Using system protoc instead of FetchContent protoc. Version mismatch may occur.")
    else()
      message(FATAL_ERROR "protoc not found. Please rebuild the project with CMake.")
    endif()
  endif()

  set(GRPC_PLUGIN_TARGET "")
  if(TARGET grpc_cpp_plugin)
    set(GRPC_PLUGIN_TARGET grpc_cpp_plugin)
  endif()

  if(NOT GRPC_PLUGIN_TARGET)
    find_program(GRPC_CPP_PLUGIN
      NAMES grpc_cpp_plugin
      PATHS ${CMAKE_BINARY_DIR}/_deps/grpc-build
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT GRPC_PLUGIN_TARGET AND NOT GRPC_CPP_PLUGIN)
    message(FATAL_ERROR "grpc_cpp_plugin not found in gRPC build. Please rebuild the project.")
  endif()
endif()

if(PROTOC_EXECUTABLE)
  set(Protobuf_PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE})
endif()

# mimalloc
FetchContent_Declare(
  mimalloc
  GIT_REPOSITORY https://github.com/microsoft/mimalloc
  GIT_TAG v2.1.9
)
set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MI_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mimalloc)

# yaml-cpp: try to find, otherwise fetch
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
  )
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(yaml-cpp)
endif()

# RocksDB: try to find, otherwise fetch
find_package(RocksDB QUIET)
if(NOT RocksDB_FOUND)
  FetchContent_Declare(
    rocksdb
    GIT_REPOSITORY https://github.com/facebook/rocksdb.git
    GIT_TAG v8.11.5
  )
  set(WITH_TESTS OFF CACHE BOOL "" FORCE)
  set(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "" FORCE)
  set(WITH_TOOLS OFF CACHE BOOL "" FORCE)
  set(WITH_CORE_TOOLS OFF CACHE BOOL "" FORCE)
  set(WITH_ALL_TESTS OFF CACHE BOOL "" FORCE)
  set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
  set(ROCKSDB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(FAIL_ON_WARNINGS OFF CACHE BOOL "" FORCE)
  set(PORTABLE ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(rocksdb)
  if(NOT TARGET RocksDB::rocksdb)
    add_library(RocksDB::rocksdb ALIAS rocksdb)
  endif()
endif()

# Proto generation
set(PROTO_FILE proto/ann.proto)
get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)

set(PROTO_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO_NAME}.pb.cc)
set(PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO_NAME}.pb.h)
set(GRPC_SRCS  ${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO_NAME}.grpc.pb.cc)
set(GRPC_HDRS  ${CMAKE_CURRENT_SOURCE_DIR}/proto/${PROTO_NAME}.grpc.pb.h)

set(PROTOC_DEPS "")
set(GRPC_PLUGIN_DEPS "")

if(PROTOC_TARGET)
  set(PROTOC_CMD $<TARGET_FILE:${PROTOC_TARGET}>)
  list(APPEND PROTOC_DEPS ${PROTOC_TARGET})
else()
  set(PROTOC_CMD ${PROTOC_EXECUTABLE})
endif()

if(GRPC_PLUGIN_TARGET)
  set(GRPC_PLUGIN_CMD $<TARGET_FILE:${GRPC_PLUGIN_TARGET}>)
  list(APPEND GRPC_PLUGIN_DEPS ${GRPC_PLUGIN_TARGET})
else()
  set(GRPC_PLUGIN_CMD ${GRPC_CPP_PLUGIN})
endif()

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PROTOC_CMD}
    ARGS --cpp_out=. -I. proto/ann.proto
    DEPENDS ${PROTO_FILE} ${PROTOC_DEPS}
    COMMENT "Generating protobuf C++ code from ${PROTO_FILE}"
)

if(GRPC_PLUGIN_TARGET)
  add_custom_command(
      OUTPUT ${GRPC_SRCS} ${GRPC_HDRS}
      COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PROTOC_CMD}
      ARGS --grpc_out=. --plugin=protoc-gen-grpc=$<TARGET_FILE:${GRPC_PLUGIN_TARGET}>
           -I. proto/ann.proto
      DEPENDS ${PROTO_FILE} ${PROTO_HDRS} ${PROTOC_DEPS} ${GRPC_PLUGIN_DEPS}
      COMMENT "Generating gRPC C++ code from ${PROTO_FILE}"
  )
else()
  add_custom_command(
      OUTPUT ${GRPC_SRCS} ${GRPC_HDRS}
      COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR} ${PROTOC_CMD}
      ARGS --grpc_out=. --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
           -I. proto/ann.proto
      DEPENDS ${PROTO_FILE} ${PROTO_HDRS} ${PROTOC_DEPS} ${GRPC_PLUGIN_DEPS}
      COMMENT "Generating gRPC C++ code from ${PROTO_FILE}"
  )
endif()

add_custom_target(proto_gen DEPENDS ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS})

add_library(proto_lib STATIC ${PROTO_SRCS} ${GRPC_SRCS})
add_dependencies(proto_lib proto_gen)
target_include_directories(proto_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(proto_lib PUBLIC protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection)

# Source targets
add_subdirectory(src)

add_executable(HorizANN_starter start.cpp)
target_compile_options(HorizANN_starter PRIVATE -O3 -march=native -g)
target_link_libraries(HorizANN_starter ann_config proto_lib mimalloc pthread)

# Test targets (off by default; enable with -DBUILD_TESTS=ON)
option(BUILD_TESTS "Build test targets" OFF)
if(BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/0934b7e112354d609133d2c5f973c402c9efc9b9.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(test)
endif()
