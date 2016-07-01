project(svm)

set(OTHER_SOURCE empty.cpp)

include_directories(libsvm)
add_library(svm ${OTHER_SOURCE})
