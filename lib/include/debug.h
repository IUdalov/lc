#pragma once

#include <iostream>

#ifdef NDEBUGw
#   define DEBUG if (false) std::cout
#else
#   define DEBUG std::cout << __FUNCTION__ << ":" << __LINE__ << " "
#endif