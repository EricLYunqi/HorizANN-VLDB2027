add_executable(master_server
    master_server.cpp
    query_router.cpp
    aggregator.cpp
)

target_compile_options(master_server PRIVATE -O3 -march=native -g)

target_link_libraries(master_server
    ann_config
    proto_lib
    mimalloc
    pthread
    yaml-cpp::yaml-cpp
    OpenMP::OpenMP_CXX
    RocksDB::rocksdb
)
