#pragma once
#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>
class ParsedFrame
{
public:
    ParsedFrame(const cv::Mat&, int dr = 20, int dtheta = 10, int too_close = 20);

    const cv::Mat& getThresh() const { return thresh; }
    const std::vector<cv::Point2i>& getPlayerContour() const { return player_contour;  }
private:
    struct AStar_Node {
        int point[2];
        std::shared_ptr<AStar_Node> prev;
        int pathlen;
        int totalcost;

        AStar_Node(int point[2], std::shared_ptr<AStar_Node> prev, int pathlen, int totalcost)
            : point {point[0], point[1]}
            , prev(prev)
            , pathlen(pathlen)
            , totalcost(totalcost)
        {}
    };

    enum class GridVals {
        GRID_OPEN,
        GRID_OOB,
        GRID_BLOCKED
    };

    cv::Mat frame;
    cv::Mat thresh;
    int dr;
    int dtheta;
    int too_close;

    cv::Point2i center;
    std::vector<cv::Point2i> player_contour;
    cv::Point2i player_point;

    void cover_ui();
    void threshold();
    void find_player();
    void cover_center();
};

