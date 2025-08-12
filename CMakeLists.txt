# # cmake_minimum_required(VERSION 3.10)
# # project(OpenCVProject)

# # set(CMAKE_CXX_STANDARD 17)

# # if(WIN32)
# #     set(OpenCV_DIR "C:/opencv/build")
# # endif()

# # find_package(OpenCV REQUIRED)
# # include_directories(${OpenCV_INCLUDE_DIRS})

# # add_executable(OpenCVExample main.cpp)
# # target_link_libraries(OpenCVExample ${OpenCV_LIBS})

# cmake_minimum_required(VERSION 3.10)
# project(OpenCVProject)

# set(CMAKE_CXX_STANDARD 17)

# # Set OpenCV path if you're on Windows
# if(WIN32)
#     set(OpenCV_DIR "C:/opencv/build")
# endif()

# find_package(OpenCV REQUIRED)

# # Add Dlib from source
# add_subdirectory("C:/Dlib" dlib_build)

# # Include only the Dlib root (⚠️ not C:/Dlib/dlib)
# include_directories(
#     ${OpenCV_INCLUDE_DIRS}
#     "C:/Dlib"
# )

# add_executable(OpenCVExample main.cpp drowsiness_detector.cpp)

# # Link OpenCV and Dlib
# target_link_libraries(OpenCVExample ${OpenCV_LIBS} dlib::dlib)






cmake_minimum_required(VERSION 3.10)
project(DrowsinessDetector)

set(CMAKE_CXX_STANDARD 17)

# Set OpenCV path if you're on Windows
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build")
endif()

find_package(OpenCV REQUIRED)

# Add Dlib from source
add_subdirectory("C:/Dlib" dlib_build)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "C:/Dlib"
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Add our include directory
)

# Collect all source files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Create executable with main.cpp and all source files
add_executable(DrowsinessDetector 
    main.cpp 
    ${SOURCES}
)

# Link OpenCV and Dlib
target_link_libraries(DrowsinessDetector 
    ${OpenCV_LIBS} 
    dlib::dlib
)

# Optional: Set output directory
set_target_properties(DrowsinessDetector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)



# Print some information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Source files found: ${SOURCES}")