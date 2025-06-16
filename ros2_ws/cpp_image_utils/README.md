# cpp_image_utils

## Overview
The `cpp_image_utils` project is a simple C++ application that utilizes OpenCV to load an image, draw a line on it, and display the modified image. This project serves as a demonstration of basic image processing techniques using OpenCV.

## Project Structure
```
cpp_image_utils
├── src
│   ├── main.cpp          # Entry point of the application
│   └── image_utils.cpp   # Implementation of image utility functions
├── include
│   └── image_utils.hpp   # Declarations of image utility functions
├── CMakeLists.txt        # CMake configuration file
└── README.md             # Project documentation
```

## Requirements
- C++11 or higher
- OpenCV library

## Building the Project
1. Ensure you have CMake and OpenCV installed on your system.
2. Navigate to the project directory:
   ```bash
   cd cpp_image_utils
   ```
3. Create a build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```bash
   cmake ..
   ```
5. Build the project:
   ```bash
   make
   ```

## Running the Application
After building the project, you can run the application using:
```bash
./cpp_image_utils
```

## Usage
The application will load an image specified in the `main.cpp` file, draw a line on it, and display the result in a window. Make sure to replace the image path in `main.cpp` with a valid image file path on your system.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.