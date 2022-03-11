#pragma once

#if _WIN32

#include "PlatformFunctions.h"

class WinPlatformFunctions : public PlatformFunctions {
public:
    virtual cv::Mat GetWindowCapture() override;
    virtual void pressKey(Keys key) override;
    virtual void releaseKey(Keys key) override;
};

#endif
