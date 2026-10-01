# Consuming a fatmap release package from CMake:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/fatmap-<version>-<platform>
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.16)
project(fatmap_consumer C CXX)

find_package(fatmap 0.8 REQUIRED)

add_executable(consumer_c main.c)
target_link_libraries(consumer_c PRIVATE fatmap::fatmap)

add_executable(consumer_cpp main.cpp)
target_link_libraries(consumer_cpp PRIVATE fatmap::fatmapxx)
