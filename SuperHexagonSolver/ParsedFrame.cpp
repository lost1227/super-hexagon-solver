#include "ParsedFrame.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


#include <opencv2/highgui.hpp>

using namespace std;
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
    find_player();
    if (player_contour.empty())
        return;
    cover_center();
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

void ParsedFrame::find_player() {
    Size2i playerThreshSize(300, 300);
    Point2i playerThreshOrigin(center.x - (playerThreshSize.width / 2), center.y - (playerThreshSize.height / 2));
    Mat playerThresh = thresh(Rect2i(playerThreshOrigin, playerThreshSize));
    vector<vector<Point>> contours;
    vector<Point> approx;
    Rect2i contourBounds;

    findContours(playerThresh, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) {
        approxPolyDP(contour, approx, 0.005 * arcLength(contour, true), true);
        contourBounds = boundingRect(approx);
        if (contourBounds.x < 15 || contourBounds.x + contourBounds.width > playerThreshSize.width - 15)
            continue;
        if (contourBounds.y < 15 || contourBounds.y + contourBounds.height > playerThreshSize.height - 15)
            continue;
        int area = contourArea(approx);
        if (area < 45 || area > 160)
            continue;
        for (auto& point : approx) {
            point += playerThreshOrigin;
        }
        player_contour = approx;
        break;
    }

    if (!player_contour.empty()) {
        contourBounds = boundingRect(player_contour);
        rectangle(thresh, contourBounds, 0, FILLED);
    }
}

void ParsedFrame::cover_center() {
    assert(!player_contour.empty());
    Point2i player_closest;
    double currSqrDist = numeric_limits<double>::infinity();
    for (auto& point : player_contour) {
        double sqrdist = pow(point.x - center.x, 2) + pow(point.y - center.y, 2);
        if (sqrdist < currSqrDist) {
            currSqrDist = sqrdist;
            player_closest = point;
        }
    }
    this->player_point = player_closest;

    double center_radius = norm(player_point - center);
    circle(thresh, center, center_radius, 0, FILLED);
}