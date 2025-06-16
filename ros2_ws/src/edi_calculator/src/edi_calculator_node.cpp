#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include "image_utils/image_utils.hpp"
#include <iostream>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

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
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("expected_hits", 10);
        RCLCPP_INFO(this->get_logger(), "edi_calculator_node started, waiting for maps...");
    }

private:
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received OccupancyGrid: %d x %d",
        msg->info.width, msg->info.height);
        std::cout << hello_world() << std::endl; // Call the function from image_utils
        auto image = occupancyGridToMat(*msg);

        // cv::imshow("Image with Line", image);
        // cv::waitKey(0); // Wait for a key press

        auto expected_hits = simulateLidarMask(image, 
            cv::Point(image.cols / 2, image.rows / 2), 360, 500);
        std::cout << "Expected hits: " << expected_hits.size() << std::endl;

        // for (size_t i = 0; i < std::min<size_t>(expected_hits.size(), 10); ++i) {
        // std::cout << "Hit " << i << ": (" << expected_hits[i].x << ", " << expected_hits[i].y << ")" << std::endl;
        // }

        auto pc2_msg = hitsToPointCloud2(expected_hits, msg->info, msg->header);
        pointcloud_pub_->publish(pc2_msg);


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

    sensor_msgs::msg::PointCloud2 hitsToPointCloud2(
        const std::vector<cv::Point>& hits,
        const nav_msgs::msg::MapMetaData& info,
        const std_msgs::msg::Header& header)
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header = header;
        cloud_msg.height = 1;
        cloud_msg.width = hits.size();
        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(hits.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

        for (const auto& pt : hits) {
            // Convert pixel to map coordinates
            float map_x = info.origin.position.x + (pt.x + 0.5) * info.resolution;
            float map_y = info.origin.position.y + (pt.y + 0.5) * info.resolution;
            *iter_x = map_x;
            *iter_y = map_y;
            *iter_z = 0.0;
            ++iter_x; ++iter_y; ++iter_z;
        }
        return cloud_msg;
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EdiCalculatorNode>());
    rclcpp::shutdown();
    return 0;
}