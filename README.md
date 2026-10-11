# HorizANN (SIGMOD 2027)

- `config`: config file for local deployment, where each worker is one process, communicating via gRPC.
- `include`: header files majorly adapted from `hnswlib`, including modified HNSW index.
- `proto`: gRPC proto definition.
- `src`: implementation of distributed logic.
- `test`: unit test files.
- `start.cpp`: a starter demo for HorizANN

## Dependencies

gRPC, RocksDB, googletest, protobuf, yaml-cpp, hnswlib, FeatureDetector.