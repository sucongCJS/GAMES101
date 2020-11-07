#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

// 转为弧度制
float radian(float rotation_angle)
{
    return rotation_angle*MY_PI/180.0f;
}

// 需要把眼睛移动-eye_pos到原点, 由于眼睛和物体要保持静止, 所以也移动-eye_pos
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 
        1, 0, 0, -eye_pos[0], 
        0, 1, 0, -eye_pos[1], 
        0, 0, 1, -eye_pos[2], 
        0, 0, 0, 1;
        
    view = translate * view;

    return view;
}

// 绕z轴旋转的变换矩阵, 不包含平移和缩放
Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    float angle = radian(rotation_angle);

    // 绕z轴, 改变两个sin(angle)的正负, 旋转的方向会变化
    // model(0,0) =  cos(angle);
    // model(0,1) = -sin(angle);
    // model(1,0) =  sin(angle);
    // model(1,1) =  cos(angle);

    // 绕y轴
    // model(0,0) =  cos(angle);
    // model(0,2) =  sin(angle);
    // model(2,0) = -sin(angle);
    // model(2,2) =  cos(angle);

    // 绕x轴
    model(1,1) =  cos(angle);
    model(1,2) = -sin(angle);
    model(2,1) =  sin(angle);
    model(2,2) =  cos(angle);

    return model;
}

/*
* 绕任意轴旋转, axis为过原点的一个轴
* */
Eigen::Matrix4f get_rotation(Vector3f axis, float rotation_angle){
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    float angle = radian(rotation_angle);

    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f N;  // n的反对称矩阵
    N <<         0, -axis.z(),  axis.y(),
          axis.z(),         0, -axis.x(),
         -axis.y(),  axis.x(),         0;

    Eigen::Matrix3f rotation_matrix =   cos(angle) * I + 
                                        (1-cos(angle)) * axis * axis.transpose() + 
                                        sin(angle)*N;

    model.block<3,3>(0,0) = rotation_matrix;  // block size of (3,3), starting as (0,0)

    return model;
}

// 透视投影矩阵
// eye_fov 垂直可视角度
// aspect_ratio = width/height
// zNear, zFar 都为正的
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    float half_height = tan(radian(eye_fov/2))*zNear;
    float half_width = aspect_ratio*half_height;
    float r =  half_width;
    float l = -half_width;
    float t =  half_height;
    float b = -half_height;
    float n = -zNear;
    float f = -zFar;

    // 正交投影的平移矩阵: 平移到原点
    Eigen::Matrix4f matrix_ortho_tranf = Eigen::Matrix4f::Identity();
    matrix_ortho_tranf(0,3) = -(r+l)/2.0f;
    matrix_ortho_tranf(1,3) = -(t+b)/2.0f;
    matrix_ortho_tranf(2,3) = -(n+f)/2.0f;

    // 正交投影的缩放矩阵: 缩放成一个canonical
    Eigen::Matrix4f matrix_ortho_scale = Eigen::Matrix4f::Identity();
    matrix_ortho_scale(0,0) = 2.0f/(r-l);
    matrix_ortho_scale(1,1) = 2.0f/(t-b);
    matrix_ortho_scale(2,2) = 2.0f/(n-f);

    // 正交投影矩阵
    Eigen::Matrix4f matrix_ortho = matrix_ortho_scale * matrix_ortho_tranf;

    // 透视投影转正交投影矩阵
    Eigen::Matrix4f matrix_persp2ortho = Eigen::Matrix4f::Zero();
    matrix_persp2ortho(0,0) = n;
    matrix_persp2ortho(1,1) = n;
    matrix_persp2ortho(2,2) = n+f;
    matrix_persp2ortho(2,3) = -n*f;
    matrix_persp2ortho(3,2) = 1;

    projection = matrix_ortho * matrix_persp2ortho * projection;
    // 如果只做matrix_persp2ortho变换，图形会被缩小

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {  // ./Rasterizer −r 20 运行程序并将三角形旋转20度,然后将结果存在output.png中
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {  // ./Rasterizer −r 20 image.png 运行程序并将三角形旋转20度，然后将结果存在image.png中
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);  // 初始化490000个像素点的rgb值

    Eigen::Vector3f eye_pos = {0, 0, 5};  // 眼睛的位置

    Eigen::Vector3f axis = {1,1,0};  // 旋转轴
    axis = {rand()%99/100.0f, rand()%99/100.0f, rand()%99/100.0f};

    std::vector<Eigen::Vector3f> pos{{0.6, 0, 0}, {0, 0.6, 0}, {-0.6, 0, 0}};  // 三角形三个点的坐标

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};  //?

    auto pos_id = r.load_positions(pos);  // ={0}
    auto ind_id = r.load_indices(ind);  // ={1}

    int key = 0;
    int frame_count = 0;

    // 如果是命令行模式, 就只是保存旋转后的三角形图片
    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));  // 旋转好角度
        r.set_model(get_rotation(axis, angle));
        r.set_view(get_view_matrix(eye_pos));  
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));  // near和far的距离够大，所以在正交投影scale时会变小，而垂直、水平可视长度小于2（r-l, t-b<2），所以scale后会变大

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    int i=0;  // for random dance

    while (key != 27) {  // esc键退出
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        // if (key == 'a') {
        //     angle += 10;
        // }
        // else if (key == 'd') {
        //     angle -= 10;
        // }

        // dance
        angle -= 10;
    }

    return 0;
}
