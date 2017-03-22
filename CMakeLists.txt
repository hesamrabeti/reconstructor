cmake_minimum_required(VERSION 3.6)
project(reconstructor)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen)

set(OpenCV_DIR "/usr/lib/opencv")
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc highgui opencv_video)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3/")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCE_FILES reconstructor.cc)
add_executable(reconstructor ${SOURCE_FILES})

target_link_libraries(reconstructor ceres pthread glog gflags ${OpenCV_LIBS})
