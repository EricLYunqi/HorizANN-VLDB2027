file(GLOB TEST_SOURCES "*.cpp")

add_executable(tests
    ${TEST_SOURCES}
    mock/MockAggregator.cpp
    mock/MockQueryRouter.cpp
    mock/MockClusterIndex.cpp
    ${CMAKE_SOURCE_DIR}/src/master/aggregator.cpp
    ${CMAKE_SOURCE_DIR}/src/master/query_router.cpp
    ${CMAKE_SOURCE_DIR}/src/master/master_server.cpp
    ${CMAKE_SOURCE_DIR}/src/worker/cluster_index.cpp
    ${CMAKE_SOURCE_DIR}/src/worker/worker_server.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

add_dependencies(tests proto_gen)
target_compile_options(tests PRIVATE -O3 -march=native -g)

# src/ added so test files can include across subdirectories (e.g. "master/aggregator.h")
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(tests
    ann_config
    mimalloc
    GTest::gtest_main
    pthread
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc++_reflection
    yaml-cpp::yaml-cpp
    OpenMP::OpenMP_CXX
    RocksDB::rocksdb
)

target_compile_definitions(tests PRIVATE ANN_BUILDING_TESTS=1)

add_test(NAME DistributedHNSWTests COMMAND tests)
