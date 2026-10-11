add_library(ann_config STATIC config_loader.cpp)

# PUBLIC so dependents (master_server, tests) inherit these include paths
target_include_directories(ann_config PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/proto
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ann_config PUBLIC yaml-cpp::yaml-cpp)
