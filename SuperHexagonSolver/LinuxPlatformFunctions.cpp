#if __linux__

#include "LinuxPlatformFunctions.h"

using namespace cv;

LinuxPlatformFunctions::LinuxPlatformFunctions() {}
LinuxPlatformFunctions::~LinuxPlatformFunctions() {}

Mat LinuxPlatformFunctions::GetWindowCapture() {
    Mat mat;
    mat.create(600, 940, CV_8UC4);
    return mat;
}

void LinuxPlatformFunctions::pressKey(Keys key) {
    // TODO
}

void LinuxPlatformFunctions::releaseKey(Keys key) {
    // TODO
}

#endif
