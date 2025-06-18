#include "image_utils/image_utils.hpp"

std::string hello_world() {
    return "Hello, world! REAL!";
}

std::vector<cv::Point> simulateLidarMask(
    const cv::Mat& map,
    const cv::Point& robot_pos,
    int n_beams,
    int max_range_px
) {
    // 1. Draw all rays on a single mask
    cv::Mat rays_mask = cv::Mat::zeros(map.size(), CV_8UC1);
    std::vector<cv::Point> ray_ends(n_beams);
    for (int i = 0; i < n_beams; ++i) {
        double angle = 2 * CV_PI * i / n_beams;
        cv::Point end(
            static_cast<int>(robot_pos.x + max_range_px * std::cos(angle)),
            static_cast<int>(robot_pos.y + max_range_px * std::sin(angle))
        );
        cv::line(rays_mask, robot_pos, end, 255, 1);
        ray_ends[i] = end;
    }

    // cv::imshow("rays_mask", rays_mask);

    // 2. Find where rays hit obstacles
    cv::Mat obstacle_mask;
    cv::threshold(map, obstacle_mask, 127, 255, cv::THRESH_BINARY); // obstacles = 255

    // cv::imshow("obstacle_mask", obstacle_mask);

    cv::Mat intersection_mask;
    cv::bitwise_and(rays_mask, obstacle_mask, intersection_mask);

    // cv::imshow("intersection_mask", intersection_mask);

    // 3. For each ray, walk along the line and find the first intersection
    std::vector<cv::Point> hits;
    for (int i = 0; i < n_beams; ++i) {
        cv::LineIterator it(intersection_mask, robot_pos, ray_ends[i]);
        for (int j = 0; j < it.count; ++j, ++it) {
            if (intersection_mask.at<uchar>(it.pos()) > 0) {
                hits.push_back(it.pos());
                break;
            }
        }
        // if (!found) {
        //     hits.push_back(ray_ends[i]);
        // }
    }
    return hits;
}