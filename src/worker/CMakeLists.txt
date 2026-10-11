add_executable(worker_server
    worker_server.cpp
    cluster_index.cpp
)

target_compile_options(worker_server PRIVATE -O3 -march=native -g)

target_link_libraries(worker_server
    ann_config
    proto_lib
    mimalloc
    pthread
    OpenMP::OpenMP_CXX
    RocksDB::rocksdb
)
