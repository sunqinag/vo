#include "camera_params.h"

CameraParams::CameraParams(std::string config_path)
{
    loadParamsFromYAML(config_path);
}

void CameraParams::loadParamsFromYAML(const std::string& config_path)
{
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        // 加载基本参数
        if (config["sensor_type"]) {
            sensor_type = config["sensor_type"].as<std::string>();
        }
        
        if (config["comment"]) {
            comment = config["comment"].as<std::string>();
        }
        
        // 加载外参 T_BS
        if (config["T_BS"] && config["T_BS"]["data"]) {
            auto t_bs_data = config["T_BS"]["data"].as<std::vector<double>>();
            if (t_bs_data.size() == 16) {
                // 先加载到Eigen矩阵
                Eigen::Matrix4d t_bs_matrix;
                t_bs_matrix << t_bs_data[0], t_bs_data[1], t_bs_data[2], t_bs_data[3],
                            t_bs_data[4], t_bs_data[5], t_bs_data[6], t_bs_data[7],
                            t_bs_data[8], t_bs_data[9], t_bs_data[10], t_bs_data[11],
                            t_bs_data[12], t_bs_data[13], t_bs_data[14], t_bs_data[15];
                // 转换为Sophus::SE3d
                T_BS = Sophus::SE3d(t_bs_matrix);
            }
        }
        
        // 加载相机参数
        if (config["rate_hz"]) {
            rate_hz = config["rate_hz"].as<double>();
        }
        
        if (config["resolution"]) {
            resolution = config["resolution"].as<std::vector<int>>();
        }
        
        if (config["camera_model"]) {
            camera_model = config["camera_model"].as<std::string>();
        }
        
        if (config["intrinsics"]) {
            intrinsics = config["intrinsics"].as<std::vector<double>>();
        }
        
        if (config["distortion_model"]) {
            distortion_model = config["distortion_model"].as<std::string>();
        }
        
        if (config["distortion_coefficients"]) {
            distortion_coefficients = config["distortion_coefficients"].as<std::vector<double>>();
        }
        
        std::cout << "Successfully loaded camera parameters from " << config_path << std::endl;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading camera parameters: " << e.what() << std::endl;
        throw;
    }
}

void CameraParams::printParams() const
{
    std::cout << "=== Camera Parameters ===" << std::endl;
    std::cout << "Sensor Type: " << sensor_type << std::endl;
    std::cout << "Comment: " << comment << std::endl;
    
    std::cout << "T_BS Matrix:" << std::endl;
    std::cout << T_BS.matrix() << std::endl;
    
    std::cout << "Rate (Hz): " << rate_hz << std::endl;
    std::cout << "Resolution: " << resolution[0] << "x" << resolution[1] << std::endl;
    std::cout << "Camera Model: " << camera_model << std::endl;
    std::cout << "Intrinsics (fu, fv, cu, cv): " << intrinsics[0] << ", " << intrinsics[1] << ", " << intrinsics[2] << ", " << intrinsics[3] << std::endl;
    std::cout << "Distortion Model: " << distortion_model << std::endl;
    std::cout << "Distortion Coefficients: ";
    for (double coeff : distortion_coefficients) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;
    std::cout << "========================" << std::endl;
}

cv::Mat CameraParams::getK() const
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        intrinsics[0], 0, intrinsics[2],
        0, intrinsics[1], intrinsics[3],
        0, 0, 1);
    return K;
}

cv::Mat CameraParams::getD() const
{
    cv::Mat D = (cv::Mat_<double>(1, distortion_coefficients.size()) << 
        distortion_coefficients[0], distortion_coefficients[1], 
        distortion_coefficients[2], distortion_coefficients[3]);
    return D;
}