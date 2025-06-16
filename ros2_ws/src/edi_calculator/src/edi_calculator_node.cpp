#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include "image_utils/image_utils.hpp"
#include <iostream>

class EdiCalculatorNode : public rclcpp::Node
{
public:
    EdiCalculatorNode()
    : Node("edi_calculator_node")
    {
        subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "og_map", 10,
            std::bind(&EdiCalculatorNode::map_callback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "edi_calculator_node started, waiting for maps...");
    }

private:
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received OccupancyGrid: %d x %d",
        msg->info.width, msg->info.height);
        std::cout << hello_world() << std::endl; // Call the function from image_utils
        auto image = occupancyGridToMat(*msg);

        auto expected_hits = simulateLidarMask(image, 
            cv::Point(image.cols / 2, image.rows / 2), 360, 200);
        std::cout << "Expected hits: " << expected_hits.size() << std::endl;

        // cv::imshow("Image with Line", image);
        // cv::waitKey(0); // Wait for a key press

    }

    cv::Mat occupancyGridToMat(const nav_msgs::msg::OccupancyGrid& grid) {
        int width = grid.info.width;
        int height = grid.info.height;

        cv::Mat image(height, width, CV_8UC1);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = x + y * width;
                int8_t value = grid.data[idx];
                uint8_t pixel;
                if (value == -1) {
                    pixel = 127; // Unknown: gray
                } else if (value == 0) {
                    pixel = 255; // Free: white
                } else {
                    pixel = 0;   // Occupied: black
                }
                image.at<uint8_t>(y, x) = pixel;
            }
        }
        return image;
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EdiCalculatorNode>());
    rclcpp::shutdown();
    return 0;
}