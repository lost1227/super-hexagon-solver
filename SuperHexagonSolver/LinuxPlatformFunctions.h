#pragma once

#if __linux__

#include "PlatformFunctions.h"
#include <X11/Xlib.h>

class LinuxPlatformFunctions : public PlatformFunctions {
public:
    LinuxPlatformFunctions();
    virtual ~LinuxPlatformFunctions();

    virtual cv::Mat GetWindowCapture() override;
    virtual void pressKey(Keys key) override;
    virtual void releaseKey(Keys key) override;

private:
    cv::Mat currFrame;
    Display *display;
    Window hexWin;
};

#endif
