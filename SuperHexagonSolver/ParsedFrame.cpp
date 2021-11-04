#include "ParsedFrame.h"

#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

ParsedFrame::ParsedFrame(const Mat& frame, int dr, int dtheta, int too_close)
    : frame(frame)
    , dr(dr)
    , dtheta(dtheta)
    , too_close(too_close)
    , center(frame.size[1] / 2, frame.size[0] / 2)
{
    assert(frame.channels() == 1);
    cover_ui();
    threshold();
}

void ParsedFrame::cover_ui() {
    double max = 0;
    minMaxLoc(frame.rowRange(70, frame.size[0]), nullptr, &max);
    rectangle(frame, Point2i(0, 0), Point2i(264, 40), max, FILLED);
    rectangle(frame, Point2i(frame.size[1] - 350, 0), Point2i(frame.size[1] - 1, 66), max, FILLED);
}

void ParsedFrame::threshold() {
    Mat blurred;
    blur(frame, blurred, Size(3, 3));
    cv::threshold(blurred, thresh, 100, 225, THRESH_BINARY | THRESH_OTSU);
}