#pragma once

#include <iostream>

#ifdef NDEBUG
#define LOGGING_ENABLED 0
#else
#define LOGGING_ENABLED 1
#endif

#define LOG for(int i = 0; i < 1; i++) std::cout << __FILE__ << ":" << __LINE__ << ":"
#define FUNCTION_LOG { LOG << __FUNCTION__ << std::endl; }
