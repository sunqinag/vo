#ifndef CAMERA_PARAMS_H
#define CAMERA_PARAMS_H
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

class CameraParams
{
private:
    // 基本参数
    std::string sensor_type;
    std::string comment;

    // 外参 T_BS (body to sensor)
    Sophus::SE3d T_BS;
    
    // 相机参数
    double rate_hz;
    std::vector<int> resolution; // [width, height]
    std::string camera_model;
    std::vector<double> intrinsics; // [fu, fv, cu, cv]
    std::string distortion_model;
    std::vector<double> distortion_coefficients;
    
    void loadParamsFromYAML(const std::string& config_path);

public:
    CameraParams(std::string config_path);
    
    // 获取参数的方法
    std::string getSensorType() const { return sensor_type; }
    std::string getComment() const { return comment; }
    Sophus::SE3d getT_BS() const { return T_BS; }
    Eigen::Matrix4d getT_BS_matrix() const { return T_BS.matrix(); }
    double getRateHz() const { return rate_hz; }
    std::vector<int> getResolution() const { return resolution; }
    int getWidth() const { return resolution[0]; }
    int getHeight() const { return resolution[1]; }
    std::string getCameraModel() const { return camera_model; }
    std::vector<double> getIntrinsics() const { return intrinsics; }
    double getFu() const { return intrinsics[0]; }
    double getFv() const { return intrinsics[1]; }
    double getCu() const { return intrinsics[2]; }
    double getCv() const { return intrinsics[3]; }
    std::string getDistortionModel() const { return distortion_model; }
    std::vector<double> getDistortionCoefficients() const { return distortion_coefficients; }
    cv::Mat getK() const;
    cv::Mat getD() const;
    // 打印参数
    void printParams() const;
};

#endif // CAMERA_PARAMS_H