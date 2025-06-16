cmake_minimum_required(VERSION 3.8)
project(image_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add_library(image_utils src/image_utils.cpp)


add_library(image_utils SHARED src/image_utils.cpp)
target_include_directories(image_utils PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>)

target_link_libraries(image_utils ${OpenCV_LIBS})

install(TARGETS image_utils
  EXPORT export_image_utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_include_directories(${OpenCV_INCLUDE_DIRS})
ament_export_libraries(image_utils)
ament_package()
