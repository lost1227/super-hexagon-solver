// SuperHexagonSolver.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <format>
#include <optional>

#include "ParsedFrame.h"
#include "PlatformFunctions.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat capture;
    Mat image;
    Mat plotted;
    Mat vis;

    chrono::high_resolution_clock::time_point timer;
    chrono::milliseconds timediff;

    string fps;

    optional<Keys> lastKey;

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
            }
            if (lastKey.has_value()) {
                pressKey(*lastKey);
            }
        }

        plotted = frame.showPlottedPath();

        vconcat(image, plotted, vis);

        timediff = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timer);
        fps = std::format("{:.2f} FPS", 1000.0 / timediff.count());

        putText(vis, fps, Point(10, vis.size[0] - 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);
        
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

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
