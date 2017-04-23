cmake_minimum_required(VERSION 2.8)
project(lc)

include_directories(. tests lib/include third-party/libsvm)

file(GLOB SOURCES 
        "lib/**/*.h"
        "lib/**/*.cpp")

file(GLOB UTILS_SOURCES
        "utils/*.cpp"
        "utils/*.h")

file(GLOB TEST_SOURCES
        "tests/*.h"
        "tests/*.cpp")

# Supress warning CMP0042
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif()

add_library(lc SHARED ${SOURCES})

add_executable(tests ${UTILS_SOURCES} ${TEST_SOURCES})
target_link_libraries(tests lc)

add_executable(lc-train ${UTILS_SOURCES} utils/cmd/lc_train.cpp)
target_link_libraries(lc-train lc)

add_executable(lc-predict ${UTILS_SOURCES} utils/cmd/lc_predict.cpp)
target_link_libraries(lc-predict lc)

add_executable(lc-generate ${UTILS_SOURCES} utils/cmd/lc_generate.cpp)
target_link_libraries(lc-generate lc)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(GENERAL "-std=c++14")
    set(BUILD_FLAVOR "-O2 -g -DNDEBUG")
    set(ENABLE_WARNINGS "-Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-overflow=5 -Wundef -Wno-unused -Wno-variadic-macros -Wno-parentheses -fdiagnostics-show-option")
    set_target_properties(lc PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(tests PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(lc-train PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(lc-predict PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(lc-generate PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
endif()
