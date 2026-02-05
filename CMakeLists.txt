cmake_minimum_required(VERSION 3.10.0)
project(VO VERSION 0.1.0 LANGUAGES C CXX)


# 寻找OpenCV包
find_package(OpenCV REQUIRED)
# 寻找yaml-cpp包
find_package(yaml-cpp REQUIRED)
# 寻找Sophus包
find_package(Sophus REQUIRED)
# 寻找Ceres包
find_package(Ceres REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS}
    "/usr/include/eigen3"
    ${YAML_CPP_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
add_library(src src/camera_params.cpp)
target_link_libraries(src ${YAML_CPP_LIBRARIES} ${Sophus_LIBRARIES} ${CERES_LIBRARIES} fmt)

add_executable(test_triangulate test/test_triangulate.cpp )

# 链接所有库到目标
target_link_libraries(test_triangulate ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} ${Sophus_LIBRARIES} ${CERES_LIBRARIES} fmt src)