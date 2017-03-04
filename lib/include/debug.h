#pragma once

#include <iostream>

#ifdef NDEBUG
#   define DEBUG if (false) std::cout
#else
#   define DEBUG std::cout << __FILE__ << ":"  << __FUNCTION__ << ":" << __LINE__ << " "
#endif