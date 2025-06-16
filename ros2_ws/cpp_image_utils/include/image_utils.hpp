// filepath: /cpp_image_utils/cpp_image_utils/include/image_utils.hpp
#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <opencv2/opencv.hpp>

void loadImageAndDrawLine(const std::string &imagePath);

cv::Mat loadImage(const std::string& imagePath);

std::vector<cv::Point> simulateLidarMask(
    const cv::Mat& map,
    const cv::Point& robot_pos,
    int n_beams,
    int max_range_px
);

cv::Point findRayHit(
    const cv::Point& origin,
    const cv::Point& end,
    const std::vector<cv::Point>& obstacles);

std::vector<cv::Point> simulateLidarBlob(
    const cv::Mat& map,
    const cv::Point& robot_pos,
    int n_beams,
    int max_range_px
);

    #endif // IMAGE_UTILS_HPP