#include "RunModes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <format>
#include <optional>
#include <memory>

#include "ParsedFrame.h"
#include "PlatformFunctions.h"

using namespace cv;
using namespace std;

constexpr int record_fps = 60;

int playRealtime(bool should_save_video) {
    Mat capture;
    Mat image;
    Mat plotted;
    Mat vis;

    chrono::high_resolution_clock::time_point now;
    chrono::high_resolution_clock::time_point last_time;
    chrono::high_resolution_clock::time_point last_record_frame_time;
    chrono::milliseconds timediff;
    chrono::milliseconds record_frame_time = chrono::milliseconds(1000 / record_fps);
    string fps;

    optional<Keys> lastKey;

    unique_ptr<VideoWriter> writer;

    if (should_save_video)
        writer = make_unique<VideoWriter>();

    last_record_frame_time = last_time = now = chrono::high_resolution_clock::now();
    while (true) {

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

        cv::vconcat(image, plotted, vis);

        now = chrono::high_resolution_clock::now();
        timediff = chrono::duration_cast<chrono::milliseconds>(now - last_time);
        last_time = now;
        fps = std::format("{:.2f} FPS", 1000.0 / timediff.count());

        cv::putText(vis, fps, Point(10, vis.size[0] - 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

        if(writer){
            if (!writer->isOpened()) {
                writer->open("out.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), record_fps, Size(vis.size[1], vis.size[0]));
            }
            timediff = chrono::duration_cast<chrono::milliseconds>(now - last_record_frame_time);
            if (timediff >= record_frame_time) {
                last_record_frame_time = now;
                writer->write(vis);
            }
        }

        cv::imshow("Hexagonical", vis);
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