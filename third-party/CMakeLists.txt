project(svm)

set(SOURCE_SVM
        libsvm/svm.h
        libsvm/svm.cpp)

include_directories(libsvm)
add_library(svm ${SOURCE_SVM})
