#pragma once
#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>

class ParsedFrame
{
public:
    ParsedFrame(const cv::Mat&);

    const cv::Mat& getThresh() const { return thresh; }

    cv::Mat showPlottedPath() const;

    bool didFindPath() const;

    enum class Direction {
        DIR_LEFT,
        DIR_RIGHT,
        DIR_OUT
    };

    Direction getNextDir() const;
private:
    struct AStar_Node {
        cv::Point2i point;
        std::shared_ptr<AStar_Node> prev;
        int pathlen;
        int totalcost;

        AStar_Node(cv::Point2i point, std::shared_ptr<AStar_Node> prev, int pathlen, int totalcost)
            : point(point)
            , prev(prev)
            , pathlen(pathlen)
            , totalcost(totalcost)
        {}

        const bool operator<(const AStar_Node& oth) const {
            return totalcost > oth.totalcost;
        }
    };

    struct GridParams {
        int dr;
        int dtheta;
        int too_close;

        int block_shadow;

        GridParams(int dr, int dtheta, int too_close, int block_shadow)
            : dr(dr)
            , dtheta(dtheta)
            , too_close(too_close)
            , block_shadow(block_shadow)
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

    int gridPrecisionLevel = -1;

    int debugThreshInfo[2];

    cv::Point2i center;
    double center_radius;
    std::vector<cv::Point2i> player_contour;
    cv::Point2i player_point;

    cv::Mat grid;
    cv::Mat_<cv::Point2i> realCoords;

    std::vector<cv::Point2i> path;

    void cover_ui();
    void threshold();
    void find_player();
    void cover_center();
    void setup_search_grid(const GridParams& params);
    void find_path();
    double estimate_cost(cv::Point2i from);
};

