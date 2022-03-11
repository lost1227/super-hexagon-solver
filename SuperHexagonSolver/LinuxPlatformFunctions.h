#pragma once

#if __linux__

#include "PlatformFunctions.h"

class LinuxPlatformFunctions : public PlatformFunctions {
public:
    LinuxPlatformFunctions();
    virtual ~LinuxPlatformFunctions();

    virtual cv::Mat GetWindowCapture() override;
    virtual void pressKey(Keys key) override;
    virtual void releaseKey(Keys key) override;
};

#endif
