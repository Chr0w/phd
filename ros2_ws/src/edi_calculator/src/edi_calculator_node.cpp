#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

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
        // TODO: Add your processing here
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