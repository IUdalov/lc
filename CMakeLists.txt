cmake_minimum_required(VERSION 2.8)
project(build-and-test)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
endif()

include_directories(. tests lib/include third-party/libsvm)
file(GLOB SOURCE_TESTS
        "tests/*.h"
        "tests/*.cpp")

file(GLOB SOURCE_UTILS
        "tests/utils/*.cpp"
        "tests/utils/*.h")

add_subdirectory(lib)
add_subdirectory(third-party)

add_executable(build-and-test ${SOURCE_UTILS} ${SOURCE_TESTS})
target_link_libraries(build-and-test lc svm)

add_executable(lc-train ${SOURCE_UTILS} misc/lc_train.cpp)
target_link_libraries(lc-train lc)

add_executable(lc-predict ${SOURCE_UTILS} misc/lc_predict.cpp)
target_link_libraries(lc-predict lc)

add_executable(lc-generate ${SOURCE_UTILS} misc/lc_generate.cpp)
target_link_libraries(lc-generate lc)
