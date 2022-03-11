#pragma once

#include <memory>
#include <opencv2/core/mat.hpp>

class PlatformFunctions {
public:
    enum class Keys {
        KEY_LEFT,
        KEY_RIGHT
    };

    virtual ~PlatformFunctions() {}

    virtual cv::Mat GetWindowCapture() = 0;
    virtual void pressKey(Keys key) = 0;
    virtual void releaseKey(Keys key) = 0;

    static std::unique_ptr<PlatformFunctions> getForCurrentPlatform();
};
