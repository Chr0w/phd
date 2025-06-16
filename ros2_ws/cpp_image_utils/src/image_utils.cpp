#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_utils.hpp"
#include <vector>
#include <cmath>

void loadImageAndDrawLine(const std::string& imagePath) {
    // Load the image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load the image." << std::endl;
        return;
    }

    // Draw a line on the image
    cv::line(image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(0, 255, 0), 5);

    // Display the image
    cv::imshow("Image with Line", image);
    cv::waitKey(0); // Wait for a key press
}


cv::Mat loadImage(const std::string& imagePath) {
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

    cv::imshow("rays_mask", rays_mask);

    // 2. Find where rays hit obstacles
    cv::Mat obstacle_mask;
    cv::threshold(map, obstacle_mask, 127, 255, cv::THRESH_BINARY_INV); // obstacles = 255

    cv::imshow("obstacle_mask", obstacle_mask);

    cv::Mat intersection_mask;
    cv::bitwise_and(rays_mask, obstacle_mask, intersection_mask);

    cv::imshow("intersection_mask", intersection_mask);

    // 3. For each ray, walk along the line and find the first intersection
    std::vector<cv::Point> hits;
    for (int i = 0; i < n_beams; ++i) {
        cv::LineIterator it(intersection_mask, robot_pos, ray_ends[i]);
        bool found = false;
        for (int j = 0; j < it.count; ++j, ++it) {
            if (intersection_mask.at<uchar>(it.pos()) > 0) {
                hits.push_back(it.pos());
                found = true;
                break;
            }
        }
        // if (!found) {
        //     hits.push_back(ray_ends[i]);
        // }
    }
    return hits;
}


// Helper: returns the closest obstacle point along a ray, or the max range endpoint
cv::Point findRayHit(
    const cv::Point& origin,
    const cv::Point& end,
    const std::vector<cv::Point>& obstacles)
{
    cv::Point hit = end;
    double min_dist = cv::norm(end - origin);

    cv::Point2f dir = end - origin;
    double ray_len = cv::norm(dir);
    if (ray_len == 0) return origin;
    dir /= ray_len;

    for (const auto& pt : obstacles) {
        cv::Point2f rel = pt - origin;
        double proj = rel.dot(dir);
        if (proj < 0 || proj > ray_len) continue; // Not on the ray segment

        // Perpendicular distance from point to ray
        double perp = std::abs(rel.cross(dir));
        if (perp > 0.707) continue; // Accept only close points (sqrt(2)/2 for 1px)

        double dist = cv::norm(rel);
        if (dist < min_dist) {
            min_dist = dist;
            hit = pt;
        }
    }
    return hit;
}

std::vector<cv::Point> simulateLidarBlob(
    const cv::Mat& map,
    const cv::Point& robot_pos,
    int n_beams,
    int max_range_px
) {
    // 1. Find all obstacle pixels (blobs)
    cv::Mat obstacle_mask;
    cv::threshold(map, obstacle_mask, 127, 255, cv::THRESH_BINARY_INV);
    std::vector<cv::Point> obstacles;
    cv::findNonZero(obstacle_mask, obstacles);

    // 2. For each ray, find the closest obstacle pixel
    std::vector<cv::Point> hits;
    for (int i = 0; i < n_beams; ++i) {
        double angle = 2 * CV_PI * i / n_beams;
        cv::Point end(
            static_cast<int>(robot_pos.x + max_range_px * std::cos(angle)),
            static_cast<int>(robot_pos.y + max_range_px * std::sin(angle))
        );
        hits.push_back(findRayHit(robot_pos, end, obstacles));
    }
    return hits;
}