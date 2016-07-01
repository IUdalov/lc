cmake_minimum_required(VERSION 2.8)
project(build-and-test)

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

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(GENERAL "-std=c++14")
    set(BUILD_FLAVOR "-O0 -g")
    set(ENABLE_WARNINGS "-Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-overflow=5 -Wundef -Wno-unused -Wno-variadic-macros -Wno-parentheses -fdiagnostics-show-option")
    set_target_properties(lc PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(lc-train PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(lc-predict PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(lc-generate PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
    set_target_properties(build-and-test PROPERTIES COMPILE_FLAGS "${ENABLE_WARNINGS} ${GENERAL} ${BUILD_FLAVOR}")
endif()
