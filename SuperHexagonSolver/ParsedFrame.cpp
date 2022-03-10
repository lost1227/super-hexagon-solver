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

#include <iostream>

using namespace std;
using namespace cv;

static constexpr double MEAN_TARGET = 53;
static constexpr int THRESHVAL = 150;

const ParsedFrame::GridParams ParsedFrame::params[] = { 
    GridParams(30, 15, 15, 40), 
    GridParams(10, 8, 5, 5),
    //GridParams(5, 5, 5, 0)
};

ParsedFrame::ParsedFrame(const Mat& frame)
    : colorFrame(frame)
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
        setup_search_grid(params);
        find_path();
        if (!path.empty()) {
            break;
        }
    }
}

void ParsedFrame::cover_ui() {
    int left_ui_width = 0;
    int right_ui_width = 0;
    int width = frame.size[1];

    while (frame.at<char>(0, left_ui_width) < 30) {
        if (left_ui_width >= width - 1) {
            left_ui_width = 0;
            break;
        }
        left_ui_width += 1;
    }
    while (frame.at<char>(50, width - right_ui_width - 1) < 30) {
        if (right_ui_width >= width - 1) {
            right_ui_width = 0;
            break;
        }
        right_ui_width += 1;
    }

    vector<Point2i> leftUiPoly = { Point2i(0, 0), Point2i(left_ui_width, 0), Point2i(left_ui_width - 29, 40), Point2i(0, 40) };
    vector<Point2i> rightUiPoly = {
        Point2i(width - (right_ui_width + 164), 0), Point2i(width, 0), Point2i(width, 65),
        Point2i(width - (right_ui_width - 2), 65), Point2i(width - (right_ui_width + 12), 40),
        Point2i(width - (right_ui_width + 135), 40)
    };

    fillConvexPoly(frame, leftUiPoly, 255);

    /*
    imshow("test", frame);
    waitKey(0);
    fillPoly(frame, vector({ rightUiPoly }), 255);
    imshow("test", frame);
    waitKey(0);
    */
}

void ParsedFrame::threshold() {
    Mat blurred;
    Mat hsv;
    Mat adjusted;
    cvtColor(colorFrame, hsv, COLOR_BGR2HSV);

    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);

    meanVal = mean(hsv_planes[2])[0];
    double bias = MEAN_TARGET - meanVal;

    convertScaleAbs(hsv_planes[2], adjusted, 1, bias);
    convertScaleAbs(adjusted, adjusted, 2, 0);

    int threshval = THRESHVAL;

    blur(adjusted, blurred, Size(3, 3));
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
                
                if (x == 0 && y == 0) {
                    circle(bgrthresh, real, 4, Scalar(0, 0, 255), FILLED);
                }
                else {
                    circle(bgrthresh, real, 2, Scalar(255, 0, 255), FILLED);
                }
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

        string dir;
        switch (getNextDir()) {
        case Direction::DIR_OUT:
            dir = "OUT";
            break;
        case Direction::DIR_LEFT:
            dir = "LEFT";
            break;
        case Direction::DIR_RIGHT:
            dir = "RIGHT";
            break;
        default:
            assert(false);
        }
        putText(bgrthresh, dir, Point2i(bgrthresh.size[1] - 300, 35), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 2);
    }

    if (!player_contour.empty()) {
        drawContours(bgrthresh, vector<vector<Point>>{ player_contour }, 0, Scalar(0, 255, 255), FILLED);
    }

    //assert(gridPrecisionLevel > 0);
    string searchInfo = std::format("Grid {}", gridPrecisionLevel);
    putText(bgrthresh, searchInfo, Point(10, bgrthresh.size[0] - 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    string threshInfo = std::format("Mean Val {:.4f}", meanVal);
    putText(bgrthresh, threshInfo, Point(10, bgrthresh.size[0] - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    return bgrthresh;
}

void ParsedFrame::setup_search_grid(const GridParams &params) {
    double corner_dist = norm(Point2i(0, 0) - center);
    Size2i gridSize((int)(360 / params.dtheta), (int)((corner_dist - center_radius) / params.dr));
    grid = Mat(gridSize, CV_8UC1);
    realCoords = Mat_<Point2i>(gridSize);

    Mat area_around;

    Point2i player_vector = player_point - center;
    double player_theta;
    if (player_vector.x == 0) {
        player_theta = (player_vector.y > 0) ? M_PI / 2 : -M_PI / 2;
    }
    else {
        player_theta = atan2(player_vector.y, player_vector.x);
    }
    player_theta *= 180.0 / M_PI;

    for (int y = 0; y < gridSize.height; ++y) {
        for (int x = 0; x < gridSize.width; ++x) {
            double r = (params.dr * y) + center_radius;
            double theta = (params.dtheta * x) - player_theta;
            
            Point2i real((int)(center.x + (r * cos(theta * M_PI / 180))),(int)(center.y - (r * sin(theta * M_PI / 180))));

            realCoords.at<Point2i>(y, x) = real;
            assert(realCoords.at<Point2i>(y, x) == real);

            if (real.x < 0 || real.x >= frame.size[1]) {
                grid.at<char>(y, x) = (char) GridVals::GRID_OOB;
                continue;
            }
            if (real.y < 0 || real.y >= frame.size[0]) {
                grid.at<char>(y, x) = (char)GridVals::GRID_OOB;
                continue;
            }

            area_around = thresh(Range(max(0, real.y - params.too_close), min(thresh.size[0] - 1, real.y + params.too_close)), Range(max(0, real.x - params.too_close), min(thresh.size[1] - 1, real.x + params.too_close)));
            if (countNonZero(area_around) > 0) {
                grid.at<char>(y, x) = (char)GridVals::GRID_BLOCKED;
                continue;
            }

            if (params.dr * y < 80) {
                bool fail = false;
                for (int i = 0; i < params.block_shadow; i++) {
                    Point2i test(real.x + (i * cos(theta * M_PI / 180)), real.y - (i * sin(theta * M_PI / 180)));
                    if (test.x < 0 || test.x >= frame.size[1])
                        break;
                    if (test.y < 0 || test.y >= frame.size[0])
                        break;
                    if (thresh.at<char>(test) != 0) {
                        fail = true;
                        break;
                    }
                }

                if (fail) {
                    grid.at<char>(y, x) = (char)GridVals::GRID_BLOCKED;
                    continue;
                }
            }

            grid.at<char>(y, x) = (char)GridVals::GRID_OPEN;
        }
    }
}

Point2i dirs[] = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
Point2i lastDst(0, 0);

double ParsedFrame::estimate_cost(Point2i from) {
    const Point2i& realPoint = realCoords.at<Point2i>(from);
    int distToOuterEdge = grid.size[0] - from.y;
    double distToLastDst = norm(lastDst - realPoint);
    double distToLeft = realPoint.x;
    double distToTop = realPoint.y;
    return  distToOuterEdge + 0.0005 * distToLeft + 0.0005 * distToTop;
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
        shared_ptr<AStar_Node> curr = make_shared<AStar_Node>(openset.top());
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
            double stepcost = 1 + 1.2 * curr->point.y;
            openset.push(AStar_Node(newpoint, curr, curr->pathlen + 1, curr->pathlen + stepcost + estimate_cost(newpoint)));
        }
    }
    if(!path.empty())
        lastDst = realCoords.at<Point2i>(path.back());

}

bool ParsedFrame::didFindPath() const {
    return !path.empty();
}

ParsedFrame::Direction ParsedFrame::getNextDir() const {
    assert(didFindPath());
    assert(path.size() >= 2);
    Point2i diff = path[1] - path[0];
    
    // Account for wrap-around
    if (diff.x > 1) {
        diff.x = -1;
    }

    if (diff.x == 0) {
        assert(diff.y > 0);
        return Direction::DIR_OUT;
    }
    else if (diff.x < 0) {
        return Direction::DIR_RIGHT;
    }
    else {
        return Direction::DIR_LEFT;
    }
}