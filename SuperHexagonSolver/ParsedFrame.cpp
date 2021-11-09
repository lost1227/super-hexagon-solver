#include "ParsedFrame.h"

#define _USE_MATH_DEFINES

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <format>
#include <queue>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;

const ParsedFrame::GridParams ParsedFrame::params[] = { GridParams(25, 10, 15), GridParams(10, 8, 5) };

const int threshmap[] = {
    58, // [0, 7)
    70, // [7, 10)
    80, // [10, 15)
    85, // [15, 20)
    90, // [20, 30)
    80, // [30, 40)
    80, // [40, 50)
    80, // [50, 60)
    80, // [60, 70)
    80, // [70, 80)
    80, // [80, 90)
    80, // [90, 100)
    70, // [100, 110)
    69, // [110, 120)
    63, // [120, 130)
    68, // [130, 140)
    80, // [140, 150)
    80, // [150, 160)
    80, // [160, 170)
    62, // [170, 180]
};

const float bins[] = { 0, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180 };

ParsedFrame::ParsedFrame(const Mat& frame, int dr, int dtheta, int too_close)
    : colorFrame(frame)
    , dr(dr)
    , dtheta(dtheta)
    , too_close(too_close)
    , center(frame.size[1] / 2, frame.size[0] / 2)
{
    assert(frame.channels() == 3);
    cvtColor(frame, this->frame, COLOR_BGR2GRAY);
    cover_ui();
    threshold();
    find_player();
    if (player_contour.empty())
        return;
    cover_center();
    for (int i = 0; i < (sizeof(params) / sizeof(ParsedFrame::GridParams)); i++) {
        const ParsedFrame::GridParams& params = ParsedFrame::params[i];
        gridPrecisionLevel = i;
        setup_search_grid(params.dr, params.dtheta, params.too_close);
        find_path();
        if (!path.empty()) {
            break;
        }
    }
}

void ParsedFrame::cover_ui() {
    double max = 0;
    minMaxLoc(frame.rowRange(70, frame.size[0]), nullptr, &max);
    rectangle(frame, Point2i(0, 0), Point2i(264, 40), max, FILLED);
    rectangle(frame, Point2i(frame.size[1] - 350, 0), Point2i(frame.size[1] - 1, 66), max, FILLED);
}

void ParsedFrame::threshold() {
    Mat searchArea = colorFrame(Range(150, 445), Range(330, 630));
    Mat blurred;
    Mat hsv;
    Mat hist;
    cvtColor(searchArea, hsv, COLOR_BGR2HSV);

    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);

    int histSize = (sizeof(bins) / sizeof(int)) - 1;
    const float* histRange[] = { bins };

    calcHist(&hsv_planes[0] , 1, 0, Mat(), hist, 1, &histSize, histRange, false, false);

    int maxIdx[2];
    minMaxIdx(hist, nullptr, nullptr, nullptr, maxIdx);

    int threshval = threshmap[maxIdx[0]];
    debugThreshInfo[0] = maxIdx[0];
    debugThreshInfo[1] = threshval;

    blur(frame, blurred, Size(3, 3));
    cv::threshold(blurred, thresh, threshval, 255, THRESH_BINARY);
}

void ParsedFrame::find_player() {
    Size2i playerThreshSize(300, 300);
    Point2i playerThreshOrigin(center.x - (playerThreshSize.width / 2), center.y - (playerThreshSize.height / 2));
    Mat playerThresh = thresh(Rect2i(playerThreshOrigin, playerThreshSize));
    vector<vector<Point>> contours;
    vector<pair<vector<Point>, int>> candidate_players;
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
        if (area < 2 || area > 160)
            continue;
        for (auto& point : approx) {
            point += playerThreshOrigin;
        }
        candidate_players.push_back(pair(move(approx), area));
    }

    auto player_contour_iter = max_element(candidate_players.begin(), candidate_players.end(), [](auto p1, auto p2) -> bool { return p1.second < p2.second; });
    if (player_contour_iter != candidate_players.end()) {
        player_contour = move(player_contour_iter->first);
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

    center_radius = norm(player_point - center);
    circle(thresh, center, center_radius, 0, FILLED);
}

Mat ParsedFrame::showPlottedPath() const {
    Mat bgrthresh;
    cvtColor(thresh, bgrthresh, COLOR_GRAY2BGR);

    if (grid.data != NULL) {
        for (int y = 0; y < grid.size[0]; ++y) {
            for (int x = 0; x < grid.size[1]; ++x) {
                if (grid.at<char>(y, x) != (char)GridVals::GRID_OPEN) {
                    continue;
                }

                Point2i real = realCoords.at<Point2i>(y, x);

                circle(bgrthresh, real, 2, Scalar(255, 0, 255), FILLED);
            }
        }
    }

    if (!path.empty()) {
        for (int i = 1; i < path.size(); ++i) {
            const Point2i& prev = path[i - 1];
            const Point2i& curr = path[i];

            const Point2i& realPrev = realCoords.at<Point2i>(prev);
            const Point2i& realCurr = realCoords.at<Point2i>(curr);

            arrowedLine(bgrthresh, realPrev, realCurr, Scalar(0, 255, 0), 3);
        }
    }

    if (!player_contour.empty()) {
        drawContours(bgrthresh, vector<vector<Point>>{ player_contour }, 0, Scalar(0, 255, 255), FILLED);
    }

    string threshInfo = std::format("Threshold {} {}", debugThreshInfo[0], debugThreshInfo[1]);
    putText(bgrthresh, threshInfo, Point(10, bgrthresh.size[0] - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    return bgrthresh;
}

void ParsedFrame::setup_search_grid(int dr, int dtheta, int too_close) {
    double corner_dist = norm(Point2i(0, 0) - center);
    Size2i gridSize((int)(360 / dtheta), (int)((corner_dist - center_radius) / dr));
    grid = Mat(gridSize, CV_8UC1);
    realCoords = Mat_<Point2i>(gridSize);

    Mat area_around;

    Point2i player_vector = player_point - center;
    double player_theta;
    if (player_vector.x == 0) {
        player_theta = (player_vector.y > 0) ? M_PI / 2 : -M_PI / 2;
    }
    else {
        player_theta = atan(player_vector.y / player_vector.x);
        if (player_vector.y < 0) {
            player_theta += M_PI;
        }
    }

    player_theta *= 180.0 / M_PI;
    for (int y = 0; y < gridSize.height; ++y) {
        for (int x = 0; x < gridSize.width; ++x) {
            double r = (dr * y) + center_radius;
            double theta = (dtheta * x) - player_theta;
            
            Point2i real((int)(center.x + (r * cos(theta * M_PI / 180))),(int)(center.y - (r * sin(theta * M_PI / 180))));

            realCoords.at<Point2i>(y, x) = real;
            assert(realCoords.at<Point2i>(y, x) == real);

            if (real.x < 0 || real.x >= frame.size[1]) {
                grid.at<char>(y, x) = (char) GridVals::GRID_OOB;
                continue;
            }
            if (real.y < 0 || real.y > frame.size[0]) {
                grid.at<char>(y, x) = (char)GridVals::GRID_OOB;
                continue;
            }

            area_around = thresh(Range(max(0, real.y - too_close), min(thresh.size[0] - 1, real.y + too_close)), Range(max(0, real.x - too_close), min(thresh.size[1] - 1, real.x + too_close)));
            if (countNonZero(area_around) > 0) {
                grid.at<char>(y, x) = (char)GridVals::GRID_BLOCKED;
                continue;
            }

            grid.at<char>(y, x) = (char)GridVals::GRID_OPEN;
        }
    }
}

Point2i dirs[] = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };

double ParsedFrame::estimate_cost(Point2i from) {
    return grid.size[0] - from.y;
}

void ParsedFrame::find_path() {
    priority_queue<AStar_Node> openset;
    auto set_comparator = [](const Point2i& p1, const Point2i& p2) -> bool {
        if (p1.y == p2.y) {
            return p1.x < p2.x;
        }
        else {
            return p1.y < p2.y;
        }
    };
    set<Point2i, decltype(set_comparator)> visited;

    path.clear();

    openset.push(AStar_Node(Point2i(0, 0), nullptr, 0, estimate_cost(Point2i(0, 0))));

    while (!openset.empty()) {
        shared_ptr<AStar_Node> curr = make_shared<AStar_Node>(move(openset.top()));
        openset.pop();
        visited.insert(curr->point);
        assert(grid.at<char>(curr->point) == grid.at<char>(curr->point.y, curr->point.x));
        if (curr->point.y == (grid.size[0] - 1) || grid.at<char>(curr->point) == (char)GridVals::GRID_OOB) {
            while (true) {
                path.push_back(curr->point);
                if (curr->prev == nullptr) {
                    break;
                }
                curr = curr->prev;
            }
            reverse(path.begin(), path.end());
            break;
        }
        for (Point2i& dir : dirs) {
            Point2i newpoint = curr->point + dir;
            newpoint.x = (newpoint.x + grid.size[1]) % grid.size[1];
            if (newpoint.y < 0) {
                continue;
            }
            if (visited.contains(newpoint)) {
                continue;
            }
            if (grid.at<char>(newpoint) == (char)GridVals::GRID_BLOCKED) {
                continue;
            }
            double stepcost = 1 + 1.2 * newpoint.y;
            openset.push(AStar_Node(newpoint, curr, curr->pathlen + 1, curr->pathlen + stepcost + estimate_cost(newpoint)));
        }
    }
}