#include "../src/camera_params.h"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

using namespace std;
using namespace cv;

shared_ptr<CameraParams> left_image_params, right_image_params;

void reduceVector(vector<cv::Point2f>& v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

vector<cv::Point2f> Undistorted(const vector<cv::Point2f>& points, const Mat& K, const Mat& D)
{
    // 去畸变后的点
    std::vector<cv::Point2f> undistorted_points;

    // 使用OpenCV的undistortPoints函数进行去畸变
    cv::undistortPoints(points, undistorted_points, K, D);

    return undistorted_points; // 返回为归一化坐标
}

void display_depth_image(std::string name,Mat image, vector<float> &depths, const vector<Point2f>& points)
{
    if (image.empty() || depths.empty() || points.empty() || depths.size() != points.size()) {
        std::cerr << "Error: Invalid input parameters for display_depth_image!" << std::endl;
        return;
    }
    
    // 创建图像副本，以便在其上绘制
    Mat depth_image = image.clone();
    
    // 找到有效深度值的范围
    float min_depth = FLT_MAX;
    float max_depth = FLT_MIN;
    for (float depth : depths) {
        if (depth > 0) { // 只考虑有效深度
            if (depth < min_depth) min_depth = depth;
            if (depth > max_depth) max_depth = depth;
        }
    }
    
    if (min_depth == FLT_MAX || max_depth == FLT_MIN) {
        std::cerr << "No valid depth values found!" << std::endl;
        return;
    }
    
    float depth_range = max_depth - min_depth;
    
    // 对每个深度值进行可视化
    for (size_t i = 0; i < depths.size(); i++) {
        float depth = depths[i];
        const Point2f& pt = points[i];
        
        if (depth > 0) { // 只处理有效深度
            // 计算颜色（从蓝色到红色的渐变）
            float normalized_depth = (depth - min_depth) / depth_range;
            int r = static_cast<int>(255 * normalized_depth);
            int g = static_cast<int>(255 * (1 - normalized_depth) * 0.7);
            int b = static_cast<int>(255 * (1 - normalized_depth));
            cv::Scalar color(b, g, r);
            
            // 确保点的位置在图像范围内
            if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows) {
                // 绘制圆形标记
                cv::circle(depth_image, pt, 3, color, -1);
                
                // 绘制深度值文本
                std::string depth_text = std::to_string(depth);
                depth_text = depth_text.substr(0, 5); // 只显示前5个字符
                cv::putText(depth_image, depth_text, cv::Point(pt.x + 8, pt.y + 5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
        }
    }
    
    // 显示深度图像
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, depth_image);
    cv::waitKey(0);
}

/**
 * @brief 三角化函数，通过左右相机的特征点计算深度值
 * 
 * @param left_pts 左相机的特征点（归一化坐标）
 * @param right_pts 右相机的特征点（归一化坐标）
 * @param left_parms 左相机参数对象
 * @param right_parms 右相机参数对象
 * @return std::vector<float> 特征点的深度值向量，0表示无效深度
 */
vector<float> Triangulate(const vector<Point2f>& left_pts, const vector<Point2f>& right_pts,
                          shared_ptr<CameraParams>& left_parms, shared_ptr<CameraParams>& right_parms)
{
    // 输入有效性检查：确保左右相机的特征点数量一致且不为空
    if (left_pts.empty() || right_pts.empty() || left_pts.size() != right_pts.size()) {
        std::cerr << "Error: Invalid input points for triangulation!" << std::endl;
        return std::vector<float>();
    }

    // 获取两个相机相对于IMU的外参（T_BS：IMU到相机的变换）
    Sophus::SE3 T_cam1InBody = left_parms->getT_BS();
    Sophus::SE3 T_cam2InBody = right_parms->getT_BS();

    // 计算第二个相机相对于第一个相机的变换矩阵
    // T_cam2Incam1 = T_cam1^(-1) * T_cam2，表示从左相机到右相机的变换
    Sophus::SE3 T_cam2Incam1 = T_cam1InBody.inverse() * T_cam2InBody;
    
    // 构建投影矩阵
    // 对于归一化坐标，左相机的投影矩阵设为[I | 0]（单位矩阵+零平移）
    cv::Mat P1 = (cv::Mat_<double>(3, 4) << 
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0);
    
    // 右相机的投影矩阵为[R | t]，其中R和t是相对于左相机的旋转和平移
    cv::Mat R, t; // 存储右相机相对于左相机的旋转矩阵和平移向量
    cv::eigen2cv(T_cam2Incam1.rotationMatrix(), R); // 将Eigen旋转矩阵转换为OpenCV矩阵
    cv::eigen2cv(T_cam2Incam1.translation(), t); // 将Eigen平移向量转换为OpenCV矩阵
    
    // 构建右相机的投影矩阵
    cv::Mat P2 = (cv::Mat_<double>(3, 4) << 
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));

    // 转换输入点格式为OpenCV triangulatePoints函数要求的2xN格式
    cv::Mat left_pts_mat(2, left_pts.size(), CV_32F);  // 左相机特征点矩阵，大小为2xN
    cv::Mat right_pts_mat(2, right_pts.size(), CV_32F); // 右相机特征点矩阵，大小为2xN
    for (size_t i = 0; i < left_pts.size(); ++i) {
        left_pts_mat.at<float>(0, i) = left_pts[i].x;  // x坐标
        left_pts_mat.at<float>(1, i) = left_pts[i].y;  // y坐标
        right_pts_mat.at<float>(0, i) = right_pts[i].x; // x坐标
        right_pts_mat.at<float>(1, i) = right_pts[i].y; // y坐标
    }

    // 执行三角化计算
    // points4D输出为4xN的齐次坐标矩阵，每列为一个特征点的[X,Y,Z,W]齐次坐标
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, left_pts_mat, right_pts_mat, points4D);

    // 处理三角化结果，提取深度值
    std::vector<float> depths;               // 存储深度值的向量
    int valid_depths = 0;                    // 有效深度值计数
    
    // 遍历所有三角化结果
    for (int i = 0; i < points4D.cols; i++)
    {
        // 提取4D齐次坐标 [X, Y, Z, W]
        double X = points4D.at<float>(0, i);
        double Y = points4D.at<float>(1, i);
        double Z = points4D.at<float>(2, i);
        double W = points4D.at<float>(3, i);

        // 检查W是否接近0，避免除以0导致的数值不稳定
        if (fabs(W) < 1e-6) {
            depths.push_back(0.0); // 无效深度
            continue;
        }

        // 归一化齐次坐标得到3D坐标 [x, y, z] = [X/W, Y/W, Z/W]
        double x_3d = X / W;
        double y_3d = Y / W;
        double z_3d = Z / W;

        // 检查深度值是否有效
        // 根据相机模型，相机可能沿着-z方向观察，因此z坐标为负表示前方
        if (z_3d < 0 && z_3d > -100.0) { // 深度在合理范围内（-100米到0）
            depths.push_back(static_cast<float>(fabs(z_3d))); // 存储绝对值作为深度值
            valid_depths++; // 有效深度计数加1
        } else {
            depths.push_back(0.0); // 无效深度
        }
    }
    
    // 输出三角化结果统计信息
    std::cout << "Triangulation: " << valid_depths << " valid depths out of " << left_pts.size() << " points" << std::endl;

    // 返回深度值向量
    return depths;
}

/**
 * @brief 检测和跟踪左右图像中的特征点
 * 
 * 该函数实现了以下功能：
 * 1. 将彩色图像转换为灰度图像
 * 2. 在左图像上检测角点特征
 * 3. 使用LK光流法在右图像上跟踪这些角点
 * 4. 过滤掉跟踪失败的点
 * 5. 可视化检测和跟踪结果
 * 
 * @param left_image 左相机彩色图像
 * @param right_image 右相机彩色图像
 * @param left_pts 输出参数，存储左图像上检测到的角点
 * @param right_pts 输出参数，存储右图像上跟踪到的对应角点
 */
void DetectAndTrackImage(Mat& left_image, Mat& right_image, vector<Point2f>& left_pts, vector<Point2f>& right_pts)
{
    vector<uchar> status; // 光流跟踪状态，1表示跟踪成功，0表示跟踪失败
    vector<float> err;    // 光流跟踪误差
    
    // 将彩色图像转换为灰度图像，光流计算需要灰度图像
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);

    // 在左图像上检测角点特征
    // 参数说明：
    // - left_gray: 输入灰度图像
    // - left_pts: 输出检测到的角点
    // - 150: 最多检测150个角点
    // - 0.3: 角点质量阈值
    // - 7: 角点之间的最小距离（像素）
    // - cv::Mat(): 感兴趣区域，这里使用整幅图像
    // - 7: 用于计算协方差矩阵的窗口大小
    // - false: 不使用Harris角点检测
    // - 0.04: 角点检测参数
    cv::goodFeaturesToTrack(left_gray, left_pts, 150, 0.3, 7, cv::Mat(), 7, false, 0.04);

    // 使用金字塔LK光流法在右图像上跟踪左图像的角点
    // 参数说明：
    // - left_gray: 前一帧图像（左图像）
    // - right_gray: 当前帧图像（右图像）
    // - left_pts: 前一帧的特征点
    // - right_pts: 当前帧的特征点（输出）
    // - status: 跟踪状态
    // - err: 跟踪误差
    // - cv::Size(21, 21): 搜索窗口大小
    // - 3: 金字塔层数
    cv::calcOpticalFlowPyrLK(left_gray, right_gray, left_pts, right_pts, status, err, cv::Size(21, 21), 3);

    // 根据跟踪状态过滤掉跟踪失败的点
    reduceVector(left_pts, status);  // 过滤左图像的角点
    reduceVector(right_pts, status); // 过滤右图像的角点

    // 可视化检测和跟踪结果
    cv::Mat left_img_copy = left_image.clone();  // 克隆左图像用于绘制
    cv::Mat right_img_copy = right_image.clone(); // 克隆右图像用于绘制

    // 创建组合图像（左右图像并排显示）
    cv::Mat combined_img;
    cv::hconcat(left_img_copy, right_img_copy, combined_img);

    // 在左图像上绘制检测到的角点（绿色）
    for (const auto& pt : left_pts)
    {
        cv::circle(combined_img, pt, 5, cv::Scalar(0, 255, 0), 2); // 绿色圆圈，半径5，线宽2
    }

    // 在右图像上绘制跟踪到的角点（红色）并连线
    int right_offset = left_image.cols; // 右图像在组合图像中的水平偏移量
    for (size_t i = 0; i < left_pts.size(); i++)
    {
        cv::Point2f right_pt = right_pts[i] + cv::Point2f(right_offset, 0); // 计算右图像点在组合图像中的位置
        cv::circle(combined_img, right_pt, 5, cv::Scalar(0, 0, 255), 2);  // 红色圆圈
        cv::line(combined_img, left_pts[i], right_pt, cv::Scalar(255, 0, 0), 1); // 蓝色连线，线宽1
    }

    // 显示检测和跟踪结果
    cv::namedWindow("Detection Results", cv::WINDOW_NORMAL); // 创建可调整大小的窗口
    cv::imshow("Detection Results", combined_img);            // 显示组合图像
    cv::waitKey(0); // 等待用户按键，0表示无限等待
}



// 定义重投影误差代价函数
struct ReprojectionError {
    ReprojectionError(const Eigen::Vector2d& observed_p1, const Eigen::Vector2d& observed_p2) 
        : observed_p1_(observed_p1), observed_p2_(observed_p2) {}

    /**
     * @brief 重投影误差代价函数的计算运算符
     * 
     * 这是ceres优化库要求的代价函数接口，用于计算特征点在第二个相机中的重投影误差。
     * 该函数将被ceres自动调用，用于优化位姿和深度参数。
     * 
     * @tparam T 模板参数，用于ceres的自动微分
     * @param q 第二个相机相对于第一个相机的旋转四元数 [qw, qx, qy, qz]
     * @param t 第二个相机相对于第一个相机的平移向量 [tx, ty, tz]
     * @param inv_depth 特征点的逆深度 (1/z)
     * @param residuals 输出的残差向量 [dx, dy]
     * @return bool 总是返回true，表示计算成功
     */
    template <typename T>
    bool operator()(const T* const q, const T* const t, const T* const inv_depth, T* residuals) const {
        // 根据逆深度计算特征点在第一个相机坐标系下的3D坐标
        // 归一化平面坐标 (x, y) 乘以深度 (1/inv_depth) 得到3D坐标 (x/z, y/z, 1/z)
        Eigen::Matrix<T, 3, 1> pt_cam1(observed_p1_(0) / inv_depth[0], observed_p1_(1) / inv_depth[0], T(1.0) / inv_depth[0]);
        
        // 将四元数转换为旋转矩阵，用于坐标变换
        Eigen::Quaternion<T> Q(q[0], q[1], q[2], q[3]);
        Eigen::Matrix<T, 3, 3> R = Q.toRotationMatrix();
        
        // 将特征点从第一个相机坐标系转换到第二个相机坐标系
        // 转换公式: pt_cam2 = R * pt_cam1 + t
        Eigen::Matrix<T, 3, 1> pt_cam2 = R * pt_cam1 + Eigen::Matrix<T, 3, 1>(t[0], t[1], t[2]);
        
        // 将3D点投影到第二个相机的归一化平面
        // 投影公式: (x/z, y/z)
        Eigen::Matrix<T, 2, 1> predicted_p2(pt_cam2(0) / pt_cam2(2), pt_cam2(1) / pt_cam2(2));
        
        // 计算重投影误差残差
        // 残差 = 预测投影位置 - 实际观测位置
        residuals[0] = predicted_p2(0) - observed_p2_(0);
        residuals[1] = predicted_p2(1) - observed_p2_(1);
        
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& observed_p1, const Eigen::Vector2d& observed_p2) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 1>(
            new ReprojectionError(observed_p1, observed_p2)));
    }

private:
    const Eigen::Vector2d observed_p1_;
    const Eigen::Vector2d observed_p2_;
};

void building_visual_constraints(vector<Point2f> &un_left_pts, vector<Point2f> &un_right_pts, vector<float> &depths,
                                 const Mat &left_image, const vector<Point2f> &left_pts)
{
    // ===========================

    double q[4]; // [qw, qx, qy, qz] - 第二个相机相对于第一个相机的旋转
    q[0] = 1.0;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    double t[3]; // [tx, ty, tz] - 第二个相机相对于第一个相机的平移
    t[0] = 0.0;
    t[1] = 0.0;
    t[2] = 0.0;

    ceres::Problem problem;

    // 存储所有特征点的逆深度，确保它们在优化过程中一直存在
    std::vector<double> inv_depths;
    inv_depths.reserve(un_left_pts.size());
    
    // 为每个特征点添加重投影误差约束
    for (size_t i = 0; i < un_left_pts.size(); ++i) {
        if (depths[i] <= 0) continue; // 跳过无效深度
        
        // 将OpenCV点转换为Eigen向量
        Eigen::Vector2d p1(un_left_pts[i].x, un_left_pts[i].y);
        Eigen::Vector2d p2(un_right_pts[i].x, un_right_pts[i].y);
        
        // 计算逆深度并存储
        inv_depths.push_back(1.0 / depths[i]);
        
        // 创建代价函数
        ceres::CostFunction* cost_function = ReprojectionError::Create(p1, p2);
        
        // 添加残差块到问题中，设置参数为待优化变量
        // 逆深度通过指针传递，指向向量中的对应元素
        problem.AddResidualBlock(cost_function, nullptr, q, t, &inv_depths.back());
        
        // 逆深度已作为优化参数，不固定
    }
    
    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    
    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 输出优化结果
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "优化后的位姿：" << std::endl;
    std::cout << "四元数 [qw, qx, qy, qz]: " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
    std::cout << "平移向量 [tx, ty, tz]: " << t[0] << " " << t[1] << " " << t[2] << std::endl;
    
    // 将四元数转换为旋转矩阵
    Eigen::Quaterniond Q(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d R = Q.toRotationMatrix();
    
    std::cout << "旋转矩阵：" << std::endl;
    std::cout << R << std::endl;
    
    // 可视化优化后的特征点深度
    vector<float> optimized_depths;
    vector<Point2f> valid_points;
    
    size_t inv_depth_idx = 0;
    for (size_t i = 0; i < un_left_pts.size(); ++i) {
        if (depths[i] <= 0) continue; // 跳过无效深度
        
        // 计算优化后的深度值（从逆深度转换）
        float optimized_depth = 1.0 / inv_depths[inv_depth_idx++];
        
        if (optimized_depth > 0) { // 只保留有效深度
            optimized_depths.push_back(optimized_depth);
            valid_points.push_back(left_pts[i]);
        }
    }
    
    // 调用display_depth_image函数进行可视化
    display_depth_image("end estimate depth",left_image, optimized_depths, valid_points);
}

int main(int, char**)
{
    std::cout << "Hello, from VO!\n";

    std::string left_image_config_path = "/media/qiangsun/新加卷/data/MH_01_easy/cam0/sensor.yaml";
    std::string right_image_config_path = "/media/qiangsun/新加卷/data/MH_01_easy/cam1/sensor.yaml";

    left_image_params = make_shared<CameraParams>(left_image_config_path);
    left_image_params->printParams();
    right_image_params = make_shared<CameraParams>(right_image_config_path);
    right_image_params->printParams();

    cv::Mat left_image = cv::imread("/media/qiangsun/新加卷/data/MH_01_easy/cam0/data/1403636579763555584.png");
    cv::Mat right_image = cv::imread("/media/qiangsun/新加卷/data/MH_01_easy/cam1/data/1403636579763555584.png");

    // 检查图像是否成功加载
    if (left_image.empty() || right_image.empty())
    {
        std::cerr << "Error: Could not load images!" << std::endl;
        return -1;
    }

    // 检测两帧图像
    std::vector<cv::Point2f> left_pts, right_pts;
    DetectAndTrackImage(left_image, right_image, left_pts, right_pts);

    // 使用内参得到归一化去畸变坐标
    vector<Point2f> undistorted_left_pts = Undistorted(left_pts, left_image_params->getK(), left_image_params->getD());
    vector<Point2f> undistorted_right_pts =
        Undistorted(right_pts, right_image_params->getK(), right_image_params->getD());

    // 三角化对应点深度
    vector<float> depths = Triangulate(undistorted_left_pts, undistorted_right_pts, left_image_params, right_image_params);

    display_depth_image("depth",left_image, depths, left_pts);
    

    building_visual_constraints(undistorted_left_pts, undistorted_right_pts, depths, left_image, left_pts);

}