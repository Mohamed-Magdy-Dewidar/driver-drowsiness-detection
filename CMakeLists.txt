# cmake_minimum_required(VERSION 3.10)
# project(OpenCVProject)

# set(CMAKE_CXX_STANDARD 17)

# if(WIN32)
#     set(OpenCV_DIR "C:/opencv/build")
# endif()

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(OpenCVExample main.cpp)
# target_link_libraries(OpenCVExample ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.10)
project(OpenCVProject)

set(CMAKE_CXX_STANDARD 17)

# Set OpenCV path if you're on Windows
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build")
endif()

find_package(OpenCV REQUIRED)

# Add Dlib from source
add_subdirectory("C:/Dlib" dlib_build)

# Include only the Dlib root (⚠️ not C:/Dlib/dlib)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "C:/Dlib"
)

add_executable(OpenCVExample main.cpp drowsiness_detector.cpp)

# Link OpenCV and Dlib
target_link_libraries(OpenCVExample ${OpenCV_LIBS} dlib::dlib)
