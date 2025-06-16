cmake_minimum_required(VERSION 3.8)
project(edi_calculator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_utils REQUIRED)

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

find_package(OpenCV REQUIRED)  # Make sure this is present

add_executable(edi_calculator_node src/edi_calculator_node.cpp)
ament_target_dependencies(edi_calculator_node rclcpp nav_msgs image_utils)
target_link_libraries(edi_calculator_node ${OpenCV_LIBS})

install(TARGETS edi_calculator_node DESTINATION lib/${PROJECT_NAME})

ament_package()
