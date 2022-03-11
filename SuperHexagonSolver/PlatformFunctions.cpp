#include "PlatformFunctions.h"

#if _WIN32

#include "WinPlatformFunctions.h"
std::unique_ptr<PlatformFunctions> PlatformFunctions::getForCurrentPlatform() {
    return std::make_unique<WinPlatformFunctions>();
}

#elif __linux__

#include "LinuxPlatformFunctions.h"
std::unique_ptr<PlatformFunctions> PlatformFunctions::getForCurrentPlatform() {
    return std::make_unique<LinuxPlatformFunctions>();
}

#endif
