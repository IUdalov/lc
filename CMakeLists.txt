cmake_minimum_required(VERSION 2.8)
project(build-and-test)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")
endif()

include_directories(. tests lib/include third-party/libsvm)
file(GLOB SOURCE_TESTS
        "tests/*.h"
        "tests/*.cpp"
        tests/utils/*.cpp)

add_subdirectory(lib)
add_subdirectory(third-party)

add_executable(build-and-test ${SOURCE_TESTS})

target_link_libraries(build-and-test lc svm)
