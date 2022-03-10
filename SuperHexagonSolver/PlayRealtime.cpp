#include "RunModes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <optional>

#include "ParsedFrame.h"
#include "PlatformFunctions.h"

using namespace cv;
using namespace std;

#define SAVE_VIDEO

int playRealtime() {
    Mat capture;
    Mat image;
    Mat plotted;
    Mat vis;

    chrono::high_resolution_clock::time_point timer;
    chrono::milliseconds timediff;

    string fps;

    optional<Keys> lastKey;

    char buff[50];

    #ifdef SAVE_VIDEO
    VideoWriter writer;
    #endif

    while (true) {
        timer = chrono::high_resolution_clock::now();

        capture = GetWindowCapture();
        if (capture.data == NULL) {
            break;
        }
        cvtColor(capture, image, COLOR_BGRA2BGR);
        ParsedFrame frame(image);

        if (lastKey.has_value()) {
            releaseKey(*lastKey);
            lastKey.reset();
        }

        if (frame.didFindPath()) {
            switch (frame.getNextDir()) {
            case ParsedFrame::Direction::DIR_LEFT:
                lastKey = Keys::KEY_LEFT;
                break;
            case ParsedFrame::Direction::DIR_RIGHT:
                lastKey = Keys::KEY_RIGHT;
                break;
            default:
                break;
            }
            if (lastKey.has_value()) {
                pressKey(*lastKey);
            }
        }

        plotted = frame.showPlottedPath();

        vconcat(image, plotted, vis);

        timediff = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timer);
        snprintf(buff, sizeof(buff), "%.2f FPS", 1000.0 / timediff.count());

        putText(vis, buff, Point(10, vis.size[0] - 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    #ifdef SAVE_VIDEO
        if (!writer.isOpened()) {
            writer.open("out.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 10, Size(vis.size[1], vis.size[0]));
        }
        writer.write(vis);
    #endif

        imshow("Hexagonical", vis);
        if (waitKey(1) == 'q') {
            break;
        }
    }
    if (lastKey.has_value()) {
        releaseKey(*lastKey);
        lastKey.reset();
    }

    return 0;
}
