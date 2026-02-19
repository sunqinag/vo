#ifndef MANAGER_H
#define MANAGER_H
#include <glog/logging.h>
#include "sensor_data.h"
#include "camera_params.h"
#include "tic_toc.h"
#include "feature_manager.h"

class Manager
{
public:
    Manager(std::string left_camera_path, std::string right_camera_path);

    void feedIMUData(IMUData &imu_data);

    void feedCameraData(CameraData &camera_data);

    cv::Mat getLeftRightMatchedImage(cv::Mat &left_image, cv::Mat &right_image);

    void undistortPoints(std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &undistorted_points);

    void triangulate(std::vector<cv::Point2f> &left_points, std::vector<cv::Point2f> &right_points,std::vector<double> &depths);

private:
    std::shared_ptr<CameraParams> left_camera_params_;
    std::shared_ptr<CameraParams> right_camera_params_;
    std::shared_ptr<FeatureManager> feature_manager_;
    int max_corner_num_ = 150;
    cv::Mat mask;

private:
    std::vector<IMUData> imu_data_buffer_;
    cv::Mat last_left_gray_;
    bool EQUALIZE=false;
    std::vector<cv::Point2f> curr_features_,curr_undistorted_features_;
    std::vector<cv::Point2f> curr_right_features_,curr_right_undistorted_features_;
    std::vector<int> feature_ids_;
    int feature_id_counter_ = 0;
    
public:
    // 获取当前特征点和ID
    std::vector<cv::Point2f> getCurrentFeatures() const { return curr_features_; }
    std::vector<cv::Point2f> getCurrentRightFeatures() const { return curr_right_features_; }
    std::vector<int> getFeatureIDs() const { return feature_ids_; }
    
    // 可视化方法
    cv::Mat visualizeFeatureTracking() const;
    cv::Mat visualizeLeftRightMatching() const;
    
    // 保存上一帧特征点，用于前后帧匹配可视化
    std::vector<cv::Point2f> prev_features_;
    std::vector<int> prev_feature_ids_;
    cv::Mat current_left_image_;
    cv::Mat current_right_image_;
};




#endif // MANAGER_H