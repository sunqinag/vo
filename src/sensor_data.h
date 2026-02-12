#ifndef SENSOR_DATA_H
#define SENSOR_DATA_H
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>


struct IMUData
{
    double timestamp;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};

struct CameraData{
    double timestamp;
    cv::Mat left_image;
    cv::Mat right_image;
};


#endif // SENSOR_DATA_H