#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_utils.hpp"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    // Initialize OpenCV
    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);

    std::cout << std::filesystem::current_path() << std::endl;

    // Load an image
    auto image_path = std::filesystem::path("/home/mircrda/phd/ros2_ws/m1.png");
    cv::Mat image = loadImage(image_path);

    auto start = std::chrono::high_resolution_clock::now();

    // Invert colors
    // cv::bitwise_not(image, image);

    cv::Point robot_pos(image.cols / 2, image.rows / 2);
    int n_beams = 360;
    int max_range_px = 200;

    auto lidar_hits = simulateLidarMask(image, robot_pos, n_beams, max_range_px);

    // Draw lidar hits
    for (const auto& pt : lidar_hits) {
        cv::circle(image, pt, 2, cv::Scalar(128), -1);
    }

    // Apply the mask to the image
    // cv::Mat maskedImage;
    // cv::bitwise_and(image, image, maskedImage, mask);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Operation took " << duration_us.count() << " milliseconds" << std::endl;

    // // Display the image
    cv::imshow("Image with Line", image);

    cv::waitKey(0); // Wait for a key press
    return 0;
}