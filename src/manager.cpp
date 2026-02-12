#include "manager.h"
#include <glog/logging.h>


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
    cv::Mat left_gray, right_gray;
    if (camera_data.left_image.channels()==3)
    {
        // 将彩色图像转换为灰度图像，光流计算需要灰度图像
        cv::cvtColor(camera_data.left_image, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(camera_data.right_image, right_gray, cv::COLOR_BGR2GRAY);
    }else{
        left_gray = camera_data.left_image;
        right_gray= camera_data.right_image;
    }

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(left_gray, left_gray);
        clahe->apply(right_gray, right_gray);
        LOG(INFO) << "CLAHE time: " << t_c.toc() << " ms";
    }
    
    // 保存当前图像
    current_left_image_ = left_gray.clone();
    current_right_image_ = right_gray.clone();
    
    // 保存当前帧的特征点和ID作为上一帧，用于后续的可视化
    std::vector<cv::Point2f> temp_prev_features = curr_features_;
    std::vector<int> temp_prev_feature_ids = feature_ids_;
    
    // 特征点检测和追踪
    if (last_left_gray_.empty())
    {
        // 第一帧，检测新特征点
        cv::goodFeaturesToTrack(left_gray, curr_features_, max_corner_num_, 0.01, 30);
        // 为新特征点分配ID
        feature_ids_.clear();
        for (size_t i = 0; i < curr_features_.size(); i++)
        {
            feature_ids_.push_back(feature_id_counter_++);
        }
        
        // 第一帧，匹配左右帧特征点
        std::vector<cv::Point2f> right_features;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(left_gray, right_gray, curr_features_, right_features, status, err);
        
        // 过滤掉匹配失败的特征点
        reduceVector(curr_features_, right_features, feature_ids_, status);
        
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
        cv::calcOpticalFlowPyrLK(last_left_gray_, left_gray, prev_features_, curr_features_, status, err);
        
        // 过滤掉追踪失败的特征点
        reduceVector(curr_features_, status);
        reduceVector(curr_right_features_, status);
        reduceVector(feature_ids_, status);
        
        // 如果特征点数量不足，检测新特征点
        if (curr_features_.size() < max_corner_num_ * 0.5)
        {
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
            
            // 为新检测的特征点匹配右帧特征点
            if (!new_features.empty())
            {
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
            }
        }
        else
        {
            // 对现有特征点进行左右帧匹配
            std::vector<cv::Point2f> right_features;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(left_gray, right_gray, curr_features_, right_features, status, err);
            
            // 过滤掉匹配失败的特征点
            reduceVector(curr_features_, right_features, feature_ids_, status);
            
            // 保存右帧特征点
            curr_right_features_ = right_features;
        }
    }
    
    // 保存上一帧特征点和ID，用于可视化
    prev_features_ = temp_prev_features;
    prev_feature_ids_ = temp_prev_feature_ids;
    
    // 特征点追踪和匹配完成
    
    // 更新上一帧图像
    last_left_gray_ = left_gray.clone();
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