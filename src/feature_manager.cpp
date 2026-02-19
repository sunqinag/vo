#include "feature_manager.h"

void FeatureManager::push_back(double timestamp, std::vector<cv::Point2f> &left_uv_norm, std::vector<cv::Point2f> &right_uv_norm, std::vector<int> feature_id,std::vector<double> depths)
{
    assert(left_uv_norm.size() == right_uv_norm.size());
    assert(right_uv_norm.size() == feature_id.size());
    
    for (int i = 0; i < feature_id.size(); i++)
    {
        auto iter = features_list_.begin();
        while (iter!=features_list_.end())
        {
            if (iter->feature_id == feature_id[i])
            {
                break;
            }
            iter++; 
        }
        
        if (iter == features_list_.end())
        {
            Feature feature(timestamp, left_uv_norm[i], right_uv_norm[i], feature_id[i],depths[i]);
            features_list_.push_back(feature);
        }
        else
        {
            iter->timestamps.push_back(timestamp);
            iter->left_uv_norm.push_back(left_uv_norm[i]);
            iter->right_uv_norm.push_back(right_uv_norm[i]);
            iter->depths.push_back(depths[i]);
        }
    }
}

