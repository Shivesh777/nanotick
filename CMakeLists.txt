cmake_minimum_required(VERSION 3.23)
project(lob_replayer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(torch REQUIRED)      # LibTorch for RL stub
find_package(benchmark REQUIRED)  # GoogleBenchmark
find_package(GTest REQUIRED)

add_subdirectory(engine)
add_subdirectory(rl)
add_subdirectory(bench)
add_subdirectory(tests)
