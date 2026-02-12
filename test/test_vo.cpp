#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>
#include <dirent.h>
#include <thread>
#include <iomanip>
#include <glog/logging.h>
#include "../src/camera_params.h"
#include "../src/sensor_data.h"
#include "../src/manager.h"

using namespace cv;
using namespace std;
using namespace Eigen;

// 全局参数：控制数据读取频率
const double IMU_FREQUENCY = 200.0;  // IMU数据读取频率 (Hz)
const double CAMERA_FREQUENCY = 20.0; // 相机数据读取频率 (Hz)

std::shared_ptr<Manager> manager;

// IMU数据回调函数
void IMUcallBack(string imu_file)
{
    std::cout << "IMU thread start" << std::endl;

    // 打开IMU数据文件
    std::ifstream imu_file_stream(imu_file);
    if (!imu_file_stream.is_open())
    {
        std::cerr << "Failed to open IMU file: " << imu_file << std::endl;
        return;
    }

    string line;
    vector<IMUData> imu_data_buffer;

    // 跳过CSV文件表头
    getline(imu_file_stream, line);

    // 逐行读取IMU数据
    while (getline(imu_file_stream, line))
    {
        istringstream iss(line);
        string token;
        vector<string> tokens;

        // 按逗号分割数据
        while (getline(iss, token, ','))
        {
            tokens.push_back(token);
        }

        // 解析IMU数据
        if (tokens.size() == 7)
        {
            IMUData imu_data;
            // 将时间戳从纳秒转换为秒
            imu_data.timestamp = stod(tokens[0]) / 1e9;
            imu_data.angular_velocity.x() = stod(tokens[1]);
            imu_data.angular_velocity.y() = stod(tokens[2]);
            imu_data.angular_velocity.z() = stod(tokens[3]);
            imu_data.linear_acceleration.x() = stod(tokens[4]);
            imu_data.linear_acceleration.y() = stod(tokens[5]);
            imu_data.linear_acceleration.z() = stod(tokens[6]);

            imu_data_buffer.push_back(imu_data);
        }
    }

    imu_file_stream.close();

    // 输出IMU数据读取结果
    cout << "Read " << imu_data_buffer.size() << " IMU measurements" << endl;
    if (!imu_data_buffer.empty())
    {
        // 设置时间戳输出格式为秒，保持9位小数精度
        cout << fixed << setprecision(9);
        cout << "First IMU timestamp: " << imu_data_buffer[0].timestamp << " s" << endl;
        cout << "Last IMU timestamp: " << imu_data_buffer.back().timestamp << " s" << endl;
        
        // 恢复正常输出格式
        cout.unsetf(ios::fixed);
        cout.precision(6);
        
        cout << "Sample IMU data:" << endl;
        cout << "  Angular velocity: [" << imu_data_buffer[0].angular_velocity.x() 
             << ", " << imu_data_buffer[0].angular_velocity.y() 
             << ", " << imu_data_buffer[0].angular_velocity.z() << "] rad/s" << endl;
        cout << "  Linear acceleration: [" << imu_data_buffer[0].linear_acceleration.x() 
             << ", " << imu_data_buffer[0].linear_acceleration.y() 
             << ", " << imu_data_buffer[0].linear_acceleration.z() << "] m/s^2" << endl;
        cout << "Simulating real-time IMU data at " << IMU_FREQUENCY << " Hz..." << endl;
        
        // 按频率输出IMU数据（模拟实时数据流）
        double delay_ms = 1000.0 / IMU_FREQUENCY;
        int output_count = 0;
        const int MAX_OUTPUT = 10; // 只输出前几个数据点
        
        for (const auto& imu_data : imu_data_buffer)
        {
            if (output_count >= MAX_OUTPUT) break;
            
            // 输出数据，时间戳为秒单位，保持9位小数精度
            cout << "IMU Data " << output_count + 1 << ": " 
                 << "Timestamp: " << fixed << setprecision(9) << imu_data.timestamp << " s, "
                 << resetiosflags(ios::fixed) << setprecision(6)
                 << "Angular velocity: [" << imu_data.angular_velocity.x() 
                 << ", " << imu_data.angular_velocity.y() 
                 << ", " << imu_data.angular_velocity.z() << "] rad/s, "
                 << "Linear acceleration: [" << imu_data.linear_acceleration.x() 
                 << ", " << imu_data.linear_acceleration.y() 
                 << ", " << imu_data.linear_acceleration.z() << "] m/s^2" << endl;
            
            output_count++;
            this_thread::sleep_for(chrono::milliseconds((int)delay_ms));
        }
    }
}

// 处理相机数据并返回左右图像文件列表
vector<pair<string, string>> loadImageFiles(string left_image_dir, string right_image_dir)
{
    vector<string> left_image_files, right_image_files;
    vector<pair<string, string>> matched_images;

    // 遍历左图像目录
    DIR* dir = opendir(left_image_dir.c_str());
    if (dir == NULL)
    {
        std::cerr << "Failed to open left image directory: " << left_image_dir << std::endl;
        return matched_images;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG) // 只处理普通文件
        {
            string filename = entry->d_name;
            if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".png")
            {
                left_image_files.push_back(left_image_dir + "/" + filename);
            }
        }
    }
    closedir(dir);

    // 遍历右图像目录
    dir = opendir(right_image_dir.c_str());
    if (dir == NULL)
    {
        std::cerr << "Failed to open right image directory: " << right_image_dir << std::endl;
        return matched_images;
    }

    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG) // 只处理普通文件
        {
            string filename = entry->d_name;
            if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".png")
            {
                right_image_files.push_back(right_image_dir + "/" + filename);
            }
        }
    }
    closedir(dir);

    // 按文件名（时间戳）排序
    sort(left_image_files.begin(), left_image_files.end());
    sort(right_image_files.begin(), right_image_files.end());

    // 匹配左右图像（假设文件名相同）
    size_t left_idx = 0, right_idx = 0;
    while (left_idx < left_image_files.size() && right_idx < right_image_files.size())
    {
        // 提取文件名（不包含路径和扩展名）
        string left_filename = left_image_files[left_idx];
        string right_filename = right_image_files[right_idx];

        size_t left_dot_pos = left_filename.find_last_of(".");
        size_t left_slash_pos = left_filename.find_last_of("/");
        string left_timestamp_str = left_filename.substr(left_slash_pos + 1, left_dot_pos - left_slash_pos - 1);

        size_t right_dot_pos = right_filename.find_last_of(".");
        size_t right_slash_pos = right_filename.find_last_of("/");
        string right_timestamp_str = right_filename.substr(right_slash_pos + 1, right_dot_pos - right_slash_pos - 1);

        // 如果时间戳相同，则读取左右图像
        if (left_timestamp_str == right_timestamp_str)
        {
            matched_images.push_back(make_pair(left_filename, right_filename));
            left_idx++;
            right_idx++;
        }
        else if (left_timestamp_str < right_timestamp_str)
        {
            left_idx++;
        }
        else
        {
            right_idx++;
        }
    }
    
    return matched_images;
}

int main(int argc, char** argv){
    // 初始化glog
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = google::INFO;
    
    LOG(INFO) << "test vo";

    // 设置数据路径（根据实际情况修改）
    std::string base_path = "/Users/sunqiang/data/mav0";
    std::string left_image_config_path = base_path + "/cam0/sensor.yaml";
    std::string right_image_config_path = base_path + "/cam1/sensor.yaml";

    manager = std::make_shared<Manager>(left_image_config_path, right_image_config_path);
    
    // 设置数据路径
    string imu_file_path = base_path + "/imu0/data.csv";
    string left_image_dir = base_path + "/cam0/data";
    string right_image_dir = base_path + "/cam1/data";
    
    // 加载图像文件列表
    vector<pair<string, string>> matched_images = loadImageFiles(left_image_dir, right_image_dir);
    LOG(INFO) << "Loaded " << matched_images.size() << " matched image pairs";
    
    // 创建并启动IMU线程
    std::thread imu_thread(IMUcallBack, imu_file_path);
    
    // 在主线程中处理相机数据和显示可视化结果
    cv::namedWindow("Feature Tracking", cv::WINDOW_NORMAL);
    cv::namedWindow("Left-Right Matching", cv::WINDOW_NORMAL);
    
    for (const auto& image_pair : matched_images)
    {
        const string& left_filename = image_pair.first;
        const string& right_filename = image_pair.second;
        
        // 提取时间戳
        size_t left_dot_pos = left_filename.find_last_of(".");
        size_t left_slash_pos = left_filename.find_last_of("/");
        string left_timestamp_str = left_filename.substr(left_slash_pos + 1, left_dot_pos - left_slash_pos - 1);
        
        CameraData camera_data;
        // 将时间戳从纳秒转换为秒
        camera_data.timestamp = stod(left_timestamp_str) / 1e9;

        // 读取图像
        camera_data.left_image = imread(left_filename, IMREAD_GRAYSCALE);
        camera_data.right_image = imread(right_filename, IMREAD_GRAYSCALE);

        // 处理相机数据（特征点追踪）
        manager->feedCameraData(camera_data);
        
        // 使用Manager类的可视化方法
        // 可视化前后帧匹配结果
        cv::Mat display_image = manager->visualizeFeatureTracking();
        cv::imshow("Feature Tracking", display_image);
        
        // 可视化左右图匹配关系
        cv::Mat left_right_image = manager->visualizeLeftRightMatching();
        cv::imshow("Left-Right Matching", left_right_image);
        
        cv::waitKey(30); // 等待30ms，模拟实时效果
    }
    
    // 关闭窗口
    cv::destroyWindow("Left-Right Matching");
    
    // 等待IMU线程完成    
    imu_thread.join();
    
    // 关闭窗口
    cv::destroyAllWindows();
    
    LOG(INFO) << "All threads completed";
    
    // 关闭glog
    google::ShutdownGoogleLogging();
    
    return 0;
}