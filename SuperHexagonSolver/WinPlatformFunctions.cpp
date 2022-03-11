#if _WIN32

#include "WinPlatformFunctions.h"

#include "Windows.h"
#include "Gdiplus.h"

#include <cassert>

using namespace cv;
using namespace std;

static BITMAPINFOHEADER createBitmapHeader(int width, int height)
{
    BITMAPINFOHEADER  bi;

    // create a bitmap
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    return bi;
}

static Mat GdiPlusScreenCapture(HWND hWnd)
{
    // get handles to a device context (DC)
    HDC hwindowDC = GetDC(hWnd);
    HDC hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

    RECT windowsize;    // get the height and width of the screen
    GetClientRect(hWnd, &windowsize);

    // define scale, height and width
    int scale = 1;
    int screenx = 0;
    int screeny = 0;
    int width = windowsize.right;
    int height = windowsize.bottom;

    Mat out;
    out.create(height, width, CV_8UC4);

    // create a bitmap
    HBITMAP hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
    BITMAPINFOHEADER bi = createBitmapHeader(width, height);

    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);
    
    StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, screenx, screeny, width, height, SRCCOPY);   //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, out.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    // avoid memory leak
    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(hWnd, hwindowDC);

    return out;
}


Mat WinPlatformFunctions::GetWindowCapture() {
    HWND hexagonWindow = FindWindowA(NULL, "Super Hexagon");
    if (hexagonWindow == NULL)
        return {};
    return GdiPlusScreenCapture(hexagonWindow);
}

void WinPlatformFunctions::pressKey(Keys key) {
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

void WinPlatformFunctions::releaseKey(Keys key) {
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

#endif
