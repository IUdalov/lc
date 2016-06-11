cmake_minimum_required(VERSION 2.8)
project(all)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")
endif()

include_directories(src/lib/include src/lib src/test third-party/test/include)
add_executable(test_lc ${SOURCE_EXE})

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(../third-party)

target_link_libraries(test_lc lc tests svm)
