#include "PlatformFunctions.h"

#ifdef _WIN32

#include "windows.h"

void pressKey(Keys key) {
    INPUT inkey;
    inkey.type = INPUT_KEYBOARD;
    switch (key) {
    case Keys::KEY_LEFT:
        inkey.ki.wVk = VK_LEFT;
        break;
    case Keys::KEY_RIGHT:
        inkey.ki.wVk = VK_RIGHT;
        break;
    default:
        assert(false);
    }

    inkey.ki.wScan = MapVirtualKeyExW(inkey.ki.wVk, MAPVK_VK_TO_VSC, 0);
    inkey.ki.dwFlags = 0;
    inkey.ki.time = 0;
    inkey.ki.dwExtraInfo = 0;

    SendInput(1, &inkey, sizeof(inkey));
}

void releaseKey(Keys key) {
    INPUT inkey;
    inkey.type = INPUT_KEYBOARD;
    switch (key) {
    case Keys::KEY_LEFT:
        inkey.ki.wVk = VK_LEFT;
        break;
    case Keys::KEY_RIGHT:
        inkey.ki.wVk = VK_RIGHT;
        break;
    default:
        assert(false);
    }

    inkey.ki.wScan = MapVirtualKeyExW(inkey.ki.wVk, MAPVK_VK_TO_VSC, 0);
    inkey.ki.dwFlags = KEYEVENTF_KEYUP;
    inkey.ki.time = 0;
    inkey.ki.dwExtraInfo = 0;

    SendInput(1, &inkey, sizeof(inkey));
}
#elif __linux__
void pressKey(Keys key) {
    // TODO
}
void releaseKey(Keys key) {
    // TODO
}
#endif
