#pragma once
#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>

class ParsedFrame
{
public:
    ParsedFrame(const cv::Mat&, int dr = 20, int dtheta = 10, int too_close = 20);

    const cv::Mat& getThresh() const { return thresh; }

    cv::Mat showPlottedPath() const;
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

    struct GridParams {
        int dr;
        int dtheta;
        int too_close;

        GridParams(int dr, int dtheta, int too_close)
            : dr(dr)
            , dtheta(dtheta)
            , too_close(too_close)
        {}
    };

    static const GridParams params[];

    enum class GridVals {
        GRID_OPEN,
        GRID_OOB,
        GRID_BLOCKED
    };

    cv::Mat frame;
    cv::Mat colorFrame;
    cv::Mat thresh;
    int dr;
    int dtheta;
    int too_close;

    int gridPrecisionLevel = -1;

    int debugThreshInfo[2];

    cv::Point2i center;
    double center_radius;
    std::vector<cv::Point2i> player_contour;
    cv::Point2i player_point;

    cv::Mat grid;
    cv::Mat_<cv::Point2i> realCoords;

    void cover_ui();
    void threshold();
    void find_player();
    void cover_center();
    void setup_search_grid(int dr, int dtheta, int too_close);
    void find_path();
};

