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
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV ---
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build")   # Update if needed
endif()
find_package(OpenCV REQUIRED)

# --- Dlib ---
add_subdirectory("C:/Dlib" dlib_build)

# --- ZeroMQ manual installation ---
set(ZMQ_ROOT "C:/ZeroMq")
set(ZMQ_INCLUDE_DIR "${ZMQ_ROOT}/include")        # libzmq headers
set(CPPZMQ_INCLUDE_DIR "${ZMQ_ROOT}/cppzmq")     # cppzmq headers
set(ZMQ_LIBRARY "${ZMQ_ROOT}/lib/libzmq-v143-mt-s-4_3_6.lib") # static lib

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "C:/Dlib"
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # project headers
    ${ZMQ_INCLUDE_DIR}                    # for zmq.h
    ${CPPZMQ_INCLUDE_DIR}                 # for zmq.hpp
)

# Collect all .cpp files in src/
file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Executable
add_executable(DrowsinessDetector
    main.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(DrowsinessDetector
    ${OpenCV_LIBS}
    dlib::dlib
    ${ZMQ_LIBRARY}      # static ZeroMQ
)

# Optional: put binary in /bin
set_target_properties(DrowsinessDetector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Debug info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ZeroMQ include dirs: ${ZMQ_INCLUDE_DIR}, ${CPPZMQ_INCLUDE_DIR}")
message(STATUS "ZeroMQ library: ${ZMQ_LIBRARY}")




