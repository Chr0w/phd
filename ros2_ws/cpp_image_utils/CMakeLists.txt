cmake_minimum_required(VERSION 3.10)
project(cpp_image_utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

find_package(OpenCV REQUIRED)

add_executable(cpp_image_utils src/main.cpp src/image_utils.cpp)
target_link_libraries(cpp_image_utils ${OpenCV_LIBS})