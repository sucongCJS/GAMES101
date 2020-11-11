// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"

constexpr double MY_PI = 3.1415926;

// 转为弧度制
float radian(float rotation_angle)
{
    return rotation_angle*MY_PI/180.0f;
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    float angle = radian(rotation_angle);

    // 绕x轴
    model(1,1) =  cos(angle);
    model(1,2) = -sin(angle);
    model(2,1) =  sin(angle);
    model(2,2) =  cos(angle);

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Copy-paste your implementation from the previous assignment.
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

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

    if (argc == 2)
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0,0,5};

    std::vector<Eigen::Vector3f> pos
            {
                    {3, 0, -6},
                    {0, 2, -2},
                    {-3, 0, -6},
                    {3.5, -1, -5},
                    {2.5, 1.5, -5},
                    {-1, 0.5, -5}
            };

    std::vector<Eigen::Vector3i> ind
            {
                    {0, 1, 2},
                    {3, 4, 5}
            };

    std::vector<Eigen::Vector3f> cols
            {
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0}
            };

    auto pos_id = r.load_positions(pos);  // 保存顶点信息, 每加载一次id++
    auto ind_id = r.load_indices(ind);  // 保存索引信息, 每加载一次id++, 加的是同一个id?
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';
    }

    return 0;
}
// clang-format on