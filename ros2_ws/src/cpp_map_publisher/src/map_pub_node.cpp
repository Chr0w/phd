#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>


using namespace std::chrono_literals;

class OccupancyGrid_Publisher : public rclcpp::Node
{
  public:
    OccupancyGrid_Publisher()
    : Node("map_publisher")
    {
      std::string map_image_path; // = "/home/mircrda/phd/ros2_ws/m1.png";
      this->declare_parameter("map_image", "");
      this->get_parameter("map_image", map_image_path);

      RCLCPP_INFO(this->get_logger(), "Trying to load map image: %s", map_image_path.c_str());
      // std::cout <<  << map_image_path << std::endl;
      auto loaded_map = cv::imread(map_image_path, cv::IMREAD_GRAYSCALE);
      if (loaded_map.empty()) {
          RCLCPP_ERROR(this->get_logger(), "Error: Could not load the map image.");
          rclcpp::shutdown();
          return;
      }

      map = loaded_map;
      og_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("og_map", 10);
      og_timer = this->create_wall_timer(1000ms, std::bind(&OccupancyGrid_Publisher::og_callback, this));
    }
  private:

    cv::Mat map;

    void og_callback()
    {
      auto occupancy_grid_msg = nav_msgs::msg::OccupancyGrid();

      occupancy_grid_msg.header.stamp = rclcpp::Clock().now();
      occupancy_grid_msg.header.frame_id = "map";

      occupancy_grid_msg.info.resolution = 0.05;

      occupancy_grid_msg.info.width = map.size().width;
      occupancy_grid_msg.info.height = map.size().height;

      occupancy_grid_msg.info.origin.position.x = -25.0;
      occupancy_grid_msg.info.origin.position.y = -25.0;
      occupancy_grid_msg.info.origin.position.z = 0.0;
      occupancy_grid_msg.info.origin.orientation.x = 0.0;
      occupancy_grid_msg.info.origin.orientation.y = 0.0;
      occupancy_grid_msg.info.origin.orientation.z = 0.0;
      occupancy_grid_msg.info.origin.orientation.w = 1.0;
      occupancy_grid_msg.data = map.reshape(1,1); // Flatten the matrix to a single row
      // occupancy_grid_msg.data = {100, 0, 0, 0, 0, 0, 0, 0, 0};

      og_pub->publish(occupancy_grid_msg);
      std::cout << "Published occupancy grid" << std::endl; 
      RCLCPP_INFO(this->get_logger(), "Published occupancy grid");

    }


    rclcpp::TimerBase::SharedPtr og_timer;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr og_pub;

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccupancyGrid_Publisher>());
    rclcpp::shutdown();
    return 0;
}


// image: mockbot_1.png
// resolution: 0.05
// origin: [24.975, 24.975, 0.0000]
// negate: 0
// occupied_thresh: 0.65
// free_thresh: 0.196
