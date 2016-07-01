project(lc)

set(SOURCE_LIB
        include/lc.h
        include/internal/cache.h
        lc.cpp
        utils.cpp
        loss_functions.cpp
        save_load.cpp)

include_directories(include)
add_library(lc ${SOURCE_LIB})
