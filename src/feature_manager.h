#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H
#include <opencv2/opencv.hpp>

struct Feature
{
    std::vector<double> timestamps;
    std::vector<cv::Point2f> left_uv_norm;
    std::vector<cv::Point2f> right_uv_norm;
    int feature_id; 
    std::vector<double> depths;
    
    Feature(double timestamp, cv::Point2f &left_uv_norm, cv::Point2f &right_uv_norm, int feature_id,double depth){     
        this->timestamps.push_back(timestamp);
        this->left_uv_norm.push_back(left_uv_norm);
        this->right_uv_norm.push_back(right_uv_norm);
        this->depths.push_back(depth);
        this->feature_id = feature_id;
        count++;
    }
private:
    int count = 0;    
};



class FeatureManager
{
public:
    FeatureManager(){};

    void push_back(double timestamp, std::vector<cv::Point2f> &left_uv_norm, std::vector<cv::Point2f> &right_uv_norm, std::vector<int> feature_id,std::vector<double> depths);

private:
    std::vector<Feature> features_list_;
};




#endif // FEATURE_MANAGER_H