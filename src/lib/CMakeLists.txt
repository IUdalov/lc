project(lc)

set(SOURCE_LIB
        include/lc.h
        lc.cpp
        utils.cpp)

include_directories(include)
add_library(lc ${SOURCE_LIB})
