#include "manager.h"
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>


void reduceVector(std::vector<cv::Point2f>& v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(std::vector<int>& v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(std::vector<cv::Point2f>& v1, std::vector<cv::Point2f>& v2, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v1.size()); i++)
    {
        if (status[i])
        {
            v1[j] = v1[i];
            v2[j] = v2[i];
            j++;
        }
    }
    v1.resize(j);
    v2.resize(j);
}

void reduceVector(std::vector<cv::Point2f>& v1, std::vector<cv::Point2f>& v2, std::vector<int>& v3, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v1.size()); i++)
    {
        if (status[i])
        {
            v1[j] = v1[i];
            v2[j] = v2[i];
            v3[j] = v3[i];
            j++;
        }
    }
    v1.resize(j);
    v2.resize(j);
    v3.resize(j);
}



Manager::Manager(std::string left_camera_path, std::string right_camera_path)
{
    left_camera_params_ = std::make_shared<CameraParams>(left_camera_path);
    right_camera_params_ = std::make_shared<CameraParams>(right_camera_path);
    feature_id_counter_ = 0;
    // 初始化可视化相关属性
    prev_features_.clear();
    prev_feature_ids_.clear();
    // 初始化特征点管理器
    feature_manager_ = std::make_shared<FeatureManager>();
}


void Manager::feedIMUData(IMUData &imu_data)
{
    imu_data_buffer_.push_back(imu_data);
    if (imu_data_buffer_.back().timestamp - imu_data_buffer_.front().timestamp > 2.0)
    {
        imu_data_buffer_.erase(imu_data_buffer_.begin());
    }
}


void Manager::feedCameraData(CameraData &camera_data)
{
    LOG(INFO) << "feed image timestamp: " << camera_data.timestamp;
    TicToc t_total; // 总耗时
    
    cv::Mat left_gray, right_gray;
    TicToc t_preprocess; // 预处理耗时
    if (camera_data.left_image.channels()==3)
    {
        // 将彩色图像转换为灰度图像，光流计算需要灰度图像
        cv::cvtColor(camera_data.left_image, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(camera_data.right_image, right_gray, cv::COLOR_BGR2GRAY);
    }else{
        left_gray = camera_data.left_image;
        right_gray= camera_data.right_image;
    }
    LOG(INFO) << "Image preprocessing cost time: " << t_preprocess.toc() << " ms";

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(left_gray, left_gray);
        clahe->apply(right_gray, right_gray);
        LOG(INFO) << "CLAHE cost time: " << t_c.toc() << " ms";
    }
    
    // 保存当前图像
    TicToc t_save;
    current_left_image_ = left_gray.clone();
    current_right_image_ = right_gray.clone();
    LOG(INFO) << "Clone current images cost time: " << t_save.toc() << " ms";
    
    // 保存当前帧的特征点和ID作为上一帧，用于后续的可视化
    std::vector<cv::Point2f> temp_prev_features = curr_features_;
    std::vector<int> temp_prev_feature_ids = feature_ids_;
    
    // 特征点检测和追踪
    TicToc t_feature;
    if (last_left_gray_.empty())
    {
        // 第一帧，检测新特征点
        TicToc t_detect;
        cv::goodFeaturesToTrack(left_gray, curr_features_, max_corner_num_, 0.01, 30);
        LOG(INFO) << "Feature detection cost time: " << t_detect.toc() << " ms";
        
        // 为新特征点分配ID
        TicToc t_id;
        feature_ids_.clear();
        for (size_t i = 0; i < curr_features_.size(); i++)
        {
            feature_ids_.push_back(feature_id_counter_++);
        }
        LOG(INFO) << "Feature ID assignment cost time: " << t_id.toc() << " ms";
        
        // 第一帧，匹配左右帧特征点
        TicToc t_match;
        std::vector<cv::Point2f> right_features;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(left_gray, right_gray, curr_features_, right_features, status, err);
        LOG(INFO) << "Left-right matching cost time: " << t_match.toc() << " ms";
        
        // 过滤掉匹配失败的特征点
        TicToc t_filter;
        reduceVector(curr_features_, right_features, feature_ids_, status);
        LOG(INFO) << "Filter matched features cost time: " << t_filter.toc() << " ms";
        
        // 使用RANSAC剔除误匹配
        if (curr_features_.size() >= 8) { // RANSAC需要至少8个点
            TicToc t_ransac;
            std::vector<uchar> inliers;
            cv::findFundamentalMat(curr_features_, right_features, inliers, cv::FM_RANSAC, 1.0, 0.99);
            reduceVector(curr_features_, right_features, feature_ids_, inliers);
            LOG(INFO) << "RANSAC cost time: " << t_ransac.toc() << " ms";
        }
        
        // 保存右帧特征点
        curr_right_features_ = right_features;
        
        // 第一帧，没有上一帧，所以prev_features_和prev_feature_ids_保持为空
        prev_features_.clear();
        prev_feature_ids_.clear();
    }
    else
    {
        // 后续帧，使用光流追踪特征点
        std::vector<cv::Point2f> prev_features_ = curr_features_;
        std::vector<cv::Point2f> prev_right_features_ = curr_right_features_;
        std::vector<uchar> status;
        std::vector<float> err;
        
        // 追踪左帧特征点
        TicToc t_track;
        cv::calcOpticalFlowPyrLK(last_left_gray_, left_gray, prev_features_, curr_features_, status, err);
        LOG(INFO) << "Feature tracking cost time: " << t_track.toc() << " ms";
        
        // 保存有效的prev_features_点，确保与curr_features_大小匹配
        std::vector<cv::Point2f> valid_prev_features;
        for (size_t i = 0; i < prev_features_.size(); i++) {
            if (status[i]) {
                valid_prev_features.push_back(prev_features_[i]);
            }
        }
        
        // 过滤掉追踪失败的特征点
        TicToc t_filter;
        reduceVector(curr_features_, status);
        reduceVector(curr_right_features_, status);
        reduceVector(feature_ids_, status);
        LOG(INFO) << "Filter tracked features cost time: " << t_filter.toc() << " ms";
        
        // 使用RANSAC剔除前后帧之间的误匹配
        if (curr_features_.size() >= 8) { // RANSAC需要至少8个点
            TicToc t_ransac;
            std::vector<uchar> inliers;
            cv::findFundamentalMat(valid_prev_features, curr_features_, inliers, cv::FM_RANSAC, 1.0, 0.99);
            reduceVector(curr_features_, inliers);
            reduceVector(curr_right_features_, inliers);
            reduceVector(feature_ids_, inliers);
            LOG(INFO) << "RANSAC for tracking cost time: " << t_ransac.toc() << " ms";
        }
        
        // 如果特征点数量不足，检测新特征点
        if (curr_features_.size() < max_corner_num_ * 0.5)
        {
            TicToc t_new_features;
            std::vector<cv::Point2f> new_features;
            // 创建掩码，避免在已有特征点附近检测新特征点
            cv::Mat mask = cv::Mat::ones(left_gray.size(), CV_8UC1);
            for (const auto& pt : curr_features_)
            {
                cv::circle(mask, pt, 30, 0, -1);
            }
            cv::goodFeaturesToTrack(left_gray, new_features, max_corner_num_ - curr_features_.size(), 0.01, 30, mask);
            
            // 为新特征点分配ID并添加到当前特征点列表
            for (const auto& pt : new_features)
            {
                curr_features_.push_back(pt);
                feature_ids_.push_back(feature_id_counter_++);
            }
            LOG(INFO) << "New feature detection cost time: " << t_new_features.toc() << " ms";
            
            // 为新检测的特征点匹配右帧特征点
                if (!new_features.empty())
                {
                    TicToc t_new_match;
                    std::vector<cv::Point2f> new_right_features;
                    std::vector<uchar> new_status;
                    std::vector<float> new_err;
                    cv::calcOpticalFlowPyrLK(left_gray, right_gray, new_features, new_right_features, new_status, new_err);
                    
                    // 保存之前追踪的有效特征点
                    std::vector<cv::Point2f> valid_prev_features;
                    std::vector<cv::Point2f> valid_prev_right_features;
                    std::vector<int> valid_prev_ids;
                    
                    for (size_t i = 0; i < prev_features_.size(); i++)
                    {
                        if (status[i])
                        {
                            valid_prev_features.push_back(curr_features_[i]);
                            valid_prev_right_features.push_back(curr_right_features_[i]);
                            valid_prev_ids.push_back(feature_ids_[i]);
                        }
                    }
                    
                    // 过滤掉匹配失败的新特征点
                    std::vector<cv::Point2f> valid_new_features;
                    std::vector<cv::Point2f> valid_new_right_features;
                    std::vector<int> valid_new_ids;
                    
                    for (size_t i = 0; i < new_features.size(); i++)
                    {
                        if (new_status[i])
                        {
                            valid_new_features.push_back(new_features[i]);
                            valid_new_right_features.push_back(new_right_features[i]);
                            // 找到对应的ID
                            auto it = std::find(curr_features_.begin(), curr_features_.end(), new_features[i]);
                            if (it != curr_features_.end())
                            {
                                size_t idx = it - curr_features_.begin();
                                valid_new_ids.push_back(feature_ids_[idx]);
                            }
                        }
                    }
                    
                    // 使用RANSAC剔除新特征点的误匹配
                    if (valid_new_features.size() >= 8) { // RANSAC需要至少8个点
                        std::vector<uchar> inliers;
                        cv::findFundamentalMat(valid_new_features, valid_new_right_features, inliers, cv::FM_RANSAC, 1.0, 0.99);
                        reduceVector(valid_new_features, valid_new_right_features, valid_new_ids, inliers);
                    }
                    
                    // 移除未匹配的新特征点
                    curr_features_.clear();
                    curr_right_features_.clear();
                    feature_ids_.clear();
                    
                    // 重新添加所有有效特征点
                    // 先添加之前追踪的特征点
                    for (size_t i = 0; i < valid_prev_features.size(); i++)
                    {
                        curr_features_.push_back(valid_prev_features[i]);
                        curr_right_features_.push_back(valid_prev_right_features[i]);
                        feature_ids_.push_back(valid_prev_ids[i]);
                    }
                    
                    // 再添加新匹配的特征点
                    for (size_t i = 0; i < valid_new_features.size(); i++)
                    {
                        curr_features_.push_back(valid_new_features[i]);
                        curr_right_features_.push_back(valid_new_right_features[i]);
                        feature_ids_.push_back(valid_new_ids[i]);
                    }
                    LOG(INFO) << "New feature matching cost time: " << t_new_match.toc() << " ms";
                }
        }
        else
        {
            // 对现有特征点进行左右帧匹配
            TicToc t_lr_match;
            std::vector<cv::Point2f> right_features;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(left_gray, right_gray, curr_features_, right_features, status, err);
            
            // 过滤掉匹配失败的特征点
            reduceVector(curr_features_, right_features, feature_ids_, status);
            
            // 使用RANSAC剔除误匹配
            if (curr_features_.size() >= 8) { // RANSAC需要至少8个点
                std::vector<uchar> inliers;
                cv::findFundamentalMat(curr_features_, right_features, inliers, cv::FM_RANSAC, 1.0, 0.99);
                reduceVector(curr_features_, right_features, feature_ids_, inliers);
            }
            
            // 保存右帧特征点
            curr_right_features_ = right_features;
            LOG(INFO) << "Left-right matching for existing features cost time: " << t_lr_match.toc() << " ms";
        }
    }
    LOG(INFO) << "Feature processing cost time: " << t_feature.toc() << " ms";
    
    // 保存上一帧特征点和ID，用于可视化
    prev_features_ = temp_prev_features;
    prev_feature_ids_ = temp_prev_feature_ids;
    
    // 特征点追踪和匹配完成
    LOG(INFO) << "Total feature tracking and matching cost time: " << t_feature.toc() << " ms";
    // 去畸变当前帧特征点
    undistortPoints(curr_features_, curr_undistorted_features_);
    // 去畸变右帧特征点
    undistortPoints(curr_right_features_, curr_right_undistorted_features_);

    // 三角化当前帧特征点
    std::vector<double> depths;
    triangulate(curr_features_,curr_right_features_,depths);

    // 更新特征管理器
    feature_manager_->push_back(camera_data.timestamp,curr_undistorted_features_,curr_right_undistorted_features_,feature_ids_,depths);
    
    // 更新上一帧图像
    TicToc t_update;
    last_left_gray_ = left_gray.clone();
    LOG(INFO) << "Update last frame cost time: " << t_update.toc() << " ms";
    
    LOG(INFO) << "Total feedCameraData cost time: " << t_total.toc() << " ms";
}

void Manager::undistortPoints(std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &undistorted_points)
{
    if (points.empty()) {
        undistorted_points.clear();
        return;
    }
    
    // 获取相机内参和畸变系数
    cv::Mat K = left_camera_params_->getK();
    cv::Mat D = left_camera_params_->getD();
    
    // 使用 OpenCV 的 undistortPoints 函数进行去畸变并转换为归一化坐标
    // 注意：undistortPoints 函数默认会返回归一化坐标（即除以焦距后的坐标）
    cv::undistortPoints(points, undistorted_points, K, D);
}

/**
 * @brief 对双目特征点进行三角化，计算深度值
 * @param left_points 左相机像素坐标系下的特征点
 * @param right_points 右相机像素坐标系下的特征点
 * @param depths 输出的深度值向量
 * @note 输入的特征点必须是像素坐标系下的原始坐标，因为投影矩阵已包含相机内参
 */
void Manager::triangulate(std::vector<cv::Point2f> &left_points,std::vector<cv::Point2f> &right_points,std::vector<double> &depths){
    depths.clear();
    if (left_points.empty() || right_points.empty() || left_points.size() != right_points.size()) {
        return;
    }
    
    // 获取左右相机的内参
    cv::Mat K_left = left_camera_params_->getK();
    cv::Mat K_right = right_camera_params_->getK();
    
    // 从相机参数中获取右相机相对于左相机的位姿
    // 左相机相对于机体的位姿
    Sophus::SE3d T_BS_left = left_camera_params_->getT_BS();
    // 右相机相对于机体的位姿
    Sophus::SE3d T_BS_right = right_camera_params_->getT_BS();
    // 右相机相对于左相机的位姿
    Sophus::SE3d T_Sleft_Sright = T_BS_left.inverse() * T_BS_right;
    
    // 提取旋转矩阵和平移向量
    Eigen::Matrix3d R_eigen = T_Sleft_Sright.rotationMatrix();
    Eigen::Vector3d t_eigen = T_Sleft_Sright.translation();
    
    // 转换为OpenCV格式
    cv::Mat R, t;
    cv::eigen2cv(R_eigen, R);
    cv::eigen2cv(t_eigen, t);
    
    // 构造投影矩阵
    cv::Mat P1 = K_left * cv::Mat::eye(3, 4, CV_64F); // 左相机投影矩阵
    cv::Mat P2 = K_right * cv::Mat(cv::Mat::eye(3, 4, CV_64F)); // 右相机投影矩阵
    P2.colRange(0, 3) = P2.colRange(0, 3) * R;
    P2.col(3) = P2.colRange(0, 3) * t + P2.col(3);
    
    // 对每个特征点进行三角化
    for (size_t i = 0; i < left_points.size(); i++) {
        // 构造特征点矩阵
        cv::Mat points4D;
        std::vector<cv::Point2f> points1 = {left_points[i]};
        std::vector<cv::Point2f> points2 = {right_points[i]};
        
        // 使用OpenCV的三角化函数
        cv::triangulatePoints(P1, P2, points1, points2, points4D);
        
        // 转换为齐次坐标
        cv::Mat point3D = points4D.col(0) / points4D.at<double>(3, 0);
        
        // 计算深度（z坐标）
        double depth = point3D.at<double>(2, 0);
        depths.push_back(depth);
    }
}

cv::Mat Manager::visualizeFeatureTracking() const
{
    cv::Mat display_image;
    if (current_left_image_.channels() == 1)
    {
        cv::cvtColor(current_left_image_, display_image, cv::COLOR_GRAY2BGR);
    }else{
        display_image = current_left_image_.clone();
    }
    
    // 绘制前后帧匹配连线
    if (!prev_features_.empty() && !curr_features_.empty())
    {
        // 基于ID匹配前后帧特征点
        for (size_t i = 0; i < curr_features_.size(); i++)
        {
            int current_id = feature_ids_[i];
            // 查找上一帧中相同ID的特征点
            auto it = find(prev_feature_ids_.begin(), prev_feature_ids_.end(), current_id);
            if (it != prev_feature_ids_.end())
            {
                size_t prev_idx = it - prev_feature_ids_.begin();
                // 绘制连线
                cv::line(display_image, prev_features_[prev_idx], curr_features_[i], cv::Scalar(255, 0, 0), 1);
            }
        }
    }
    
    // 绘制当前帧特征点和ID
    for (size_t i = 0; i < curr_features_.size(); i++)
    {
        // 绘制特征点
        cv::circle(display_image, curr_features_[i], 3, cv::Scalar(0, 255, 0), -1);
        // 绘制特征点ID
        cv::putText(display_image, std::to_string(feature_ids_[i]), curr_features_[i] + cv::Point2f(5, -5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    
    return display_image;
}

cv::Mat Manager::visualizeLeftRightMatching() const
{
    cv::Mat left_right_image;
    // 创建左右拼接图像
    cv::hconcat(current_left_image_, current_right_image_, left_right_image);
    if (left_right_image.channels() == 1)
    {
        cv::cvtColor(left_right_image, left_right_image, cv::COLOR_GRAY2BGR);
    }
    
    // 绘制左右图匹配连线
    int offset = current_left_image_.cols; // 右图像的水平偏移
    for (size_t i = 0; i < curr_features_.size() && i < curr_right_features_.size(); i++)
    {
        // 左图特征点
        cv::Point2f left_pt = curr_features_[i];
        // 右图特征点（加上偏移）
        cv::Point2f right_pt = curr_right_features_[i];
        right_pt.x += offset;
        
        // 绘制连线
        cv::line(left_right_image, left_pt, right_pt, cv::Scalar(0, 0, 255), 1);
        
        // 绘制特征点
        cv::circle(left_right_image, left_pt, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(left_right_image, right_pt, 3, cv::Scalar(0, 255, 0), -1);
        
        // 绘制特征点ID
        cv::putText(left_right_image, std::to_string(feature_ids_[i]), left_pt + cv::Point2f(5, -5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        cv::putText(left_right_image, std::to_string(feature_ids_[i]), right_pt + cv::Point2f(5, -5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    
    return left_right_image;
}