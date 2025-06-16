#pragma once
#include <string>
#include <opencv2/opencv.hpp>


std::string hello_world();

cv::Mat load_image(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load the image." << std::endl;
        return cv::Mat();
    }

    return image;
}

std::vector<cv::Point> simulateLidarMask(
    const cv::Mat& map,
    const cv::Point& robot_pos,
    int n_beams,
    int max_range_px
);