project(tests)

set(SOURCE_EXE
        utils/utils.h
        main.cpp
        utils/utils.cpp
        primitives.cpp
        easy.cpp
        iris.cpp
        easy2.cpp
        bayes.cpp
        margins.cpp
        easy3.cpp
        basic_train.cpp)


include_directories(include)
add_library(tests ${SOURCE_LIB})
