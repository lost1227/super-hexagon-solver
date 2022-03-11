#if __linux__

#include "LinuxPlatformFunctions.h"

#include <X11/Xutil.h>

using namespace cv;

static Window window_from_name_search(Display *display, Window current, char const *needle) {
    Window retval, root, parent, *children;
    unsigned children_count;
    char *name = NULL;

    /* Check if this window has the name we seek */
    if(XFetchName(display, current, &name) > 0) {
        int r = strcmp(needle, name);
        XFree(name);
        if(r == 0) {
            return current;
        }
    }

    retval = 0;

    /* If it does not: check all subwindows recursively. */
    if(0 != XQueryTree(display, current, &root, &parent, &children, &children_count)) {
    unsigned i;
    for(i = 0; i < children_count; ++i) {
        Window win = window_from_name_search(display, children[i], needle);

        if(win != 0) {
        retval = win;
        break;
        }
    }

    XFree(children);
    }

    return retval;
}


LinuxPlatformFunctions::LinuxPlatformFunctions() {
    display = XOpenDisplay(NULL);
    hexWin = window_from_name_search(display, XDefaultRootWindow(display), "Super Hexagon");
}
LinuxPlatformFunctions::~LinuxPlatformFunctions() {
    hexWin = 0;
    XCloseDisplay(display);
}

Mat LinuxPlatformFunctions::GetWindowCapture() {
    if(hexWin == 0) {
        return {};
    }
    XWindowAttributes attributes = {0};
    XGetWindowAttributes(display, hexWin, &attributes);

    int width = attributes.width;
    int height = attributes.height;

    XImage* img = XGetImage(display, hexWin, 0, 0 , width, height, AllPlanes, ZPixmap);
    int bpp = img->bits_per_pixel;
    int matType = bpp > 24 ? CV_8UC4 : CV_8UC3;

    if(currFrame.empty() || currFrame.size[0] != height || currFrame.size[1] != width || currFrame.type() != matType) {
        currFrame.create(height, width, matType);
    }

    memcpy(currFrame.data, img->data, currFrame.total() * currFrame.elemSize());

    XDestroyImage(img);

    return currFrame;
}

void LinuxPlatformFunctions::pressKey(Keys key) {
    // TODO
}

void LinuxPlatformFunctions::releaseKey(Keys key) {
    // TODO
}

#endif
