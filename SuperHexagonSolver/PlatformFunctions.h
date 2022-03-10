#pragma once

#include <opencv2/core/mat.hpp>


cv::Mat GetWindowCapture();

enum class Keys {
    KEY_LEFT,
    KEY_RIGHT
};

void pressKey(Keys key);
void releaseKey(Keys key);
