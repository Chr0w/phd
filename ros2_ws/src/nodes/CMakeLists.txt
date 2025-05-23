cmake_minimum_required(VERSION 3.8)
project(nodes)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


# Publisher
add_executable(pub_node src/publisher.cpp) # name of executable  -  path to src code
ament_target_dependencies(pub_node rclcpp std_msgs)  # Link to required libraries

# Subscriber
add_executable(sub_node src/subscriber.cpp) # name of executable  -  path to src code
ament_target_dependencies(sub_node rclcpp std_msgs)  # Link to required libraries

# 2D Laser Publisher
add_executable(laser_scan_planar_pub_node src/laser_publisher_planar.cpp) # name of executable  -  path to src code
ament_target_dependencies(laser_scan_planar_pub_node rclcpp std_msgs sensor_msgs)  # Link to required libraries


# Needed for ROS to find the executables
install(TARGETS
  pub_node
  sub_node
  laser_scan_planar_pub_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
