#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

// 转为弧度制
float radian(float rotation_angle)
{
    return rotation_angle*MY_PI/180.0f;
}

// 欧式距离的平方
float get_euclidean_distance_square(Vector3f p1, Vector3f p2)
{
    // return ((p1-p2).array() * (p1-p2).array()).sum();
    return (p1-p2).dot(p1-p2);
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

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float half_height = tan(radian(eye_fov/2))*zNear;
    float half_width = aspect_ratio*half_height;
    float r = -half_width;
    float l =  half_width;
    float t = -half_height;
    float b =  half_height;
    float n = zNear;
    float f = zFar;

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

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColorBilinear(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;  // 高光范围， 值越大， 高光范围越小

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        float r2 = get_euclidean_distance_square(light.position, point);  // 光源到平面的距离的平方
        Eigen::Vector3f l = (light.position - point).normalized();  // light vector 从观察点指向点光源的单位向量 只保留方向信息
        Eigen::Vector3f v = (eye_pos - point).normalized();  // 从观察点指向眼睛的单位向量
        Eigen::Vector3f h = (v + l).normalized();  // halfway vector 半程向量 单位化, 只保留方向信息
        
        Eigen::Vector3f ambient  = ka.array() * amb_light_intensity.array();  // shape = (3,1) 与光源无关! 但是光源多的话, 环境光也应该亮
        Eigen::Vector3f diffuse  = kd.array() * (light.intensity / r2).array() * std::max(0.f, normal.dot(l));
        Eigen::Vector3f specular = ks.array() * (light.intensity / r2).array() * std::pow(std::max(0.f, normal.dot(h)), p);

        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}

// Blinn-Phong Reflection Model
Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);  // 环境光系数
    Eigen::Vector3f kd = payload.color;  // 漫反射系数
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);  // 镜面反射系数

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};  // 环境光强度
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* components are. Then, accumulate that result on the *result_color* object.
        float r2 = get_euclidean_distance_square(light.position, point);
        Eigen::Vector3f l = (light.position - point).normalized();
        Eigen::Vector3f v = (eye_pos - point).normalized();
        Eigen::Vector3f h = (v + l).normalized();

        Eigen::Vector3f ambient  = ka.array() * amb_light_intensity.array();  // shape = (3,1)
        Eigen::Vector3f diffuse  = kd.array() * (light.intensity / r2).array() * std::max(0.f, normal.dot(l));
        Eigen::Vector3f specular = ks.array() * (light.intensity / r2).array() * std::pow(std::max(0.f, normal.dot(h)), p);

        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = c1 * c2 * (h(u+1/w,v)-h(u,v))
    // dV = c1 * c2 * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    float c1 = 0.2, c2 = 0.1;  // 常数, 定义凹凸贴图的影响程度

    Eigen::Vector3f n = payload.normal;

    Eigen::Vector3f t(n.x() * n.y() / std::sqrt(n.x() * n.x() + n.z() * n.z()),
                      std::sqrt(n.x() * n.x() + n.z() * n.z()),
                      n.z() * n.y() / std::sqrt(n.x() * n.x() + n.z() * n.z()));
    Eigen::Vector3f b = n.cross(t);
    Eigen::Matrix3f TBN;
    TBN<<t, b, n;

    float u = payload.tex_coords(0);
    float v = payload.tex_coords(1);
    float h = payload.texture->height;
    float w = payload.texture->width;
    float dU = c1 * c2 * (payload.texture->getColor(u + 1.0/w, v)[0] - payload.texture->getColor(u, v)[0]);  // 助教说：正规的凹凸纹理应该是只有一维参量的灰度图，而本课程为了框架使用的简便性而使用了一张 RGB 图作为凹凸纹理的贴图，因此需要指定一种规则将彩色投影到灰度，而我只是「恰好」选择了 norm 而已。为了确保你们的结果与我一致，我才要求你们都使用 norm 作为计算方法。
    float dV = c1 * c2 * (payload.texture->getColor(u, v + 1.0/h)[0] - payload.texture->getColor(u, v)[0]);
    Eigen::Vector3f ln(-dU, -dV, 1);
    n = (TBN * ln).normalized();

    Eigen::Vector3f result_color(0, 0, 0);
    result_color = n;  // n 用的是纹理的颜色去做变化, 所以结果会有纹理的颜色, displacement只用了纹理去做点的移动, 最后颜色是通过blinn phong产生的, 所以没有纹理的颜色

    // 这里就不加blinn phong了

    return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)
    float c1 = 0.2, c2 = 0.1;  // 常数, 定义凹凸贴图的影响程度

    Eigen::Vector3f n = payload.normal;

    Eigen::Vector3f t(n.x() * n.y() / std::sqrt(n.x() * n.x() + n.z() * n.z()),
                      std::sqrt(n.x() * n.x() + n.z() * n.z()),
                      n.z() * n.y() / std::sqrt(n.x() * n.x() + n.z() * n.z()));
    Eigen::Vector3f b = n.cross(t);
    Eigen::Matrix3f TBN;
    TBN<<t, b, n;

    float u = payload.tex_coords(0);
    float v = payload.tex_coords(1);
    float h = payload.texture->height;
    float w = payload.texture->width;
    float dU = c1 * c2 * (payload.texture->getColor(u + 1.0/w, v).norm() - payload.texture->getColor(u, v).norm());  // 助教说：正规的凹凸纹理应该是只有一维参量的灰度图，而本课程为了框架使用的简便性而使用了一张 RGB 图作为凹凸纹理的贴图，因此需要指定一种规则将彩色投影到灰度，而我只是「恰好」选择了 norm 而已。为了确保你们的结果与我一致，我才要求你们都使用 norm 作为计算方法。
    float dV = c2 * c2 * (payload.texture->getColor(u, v + 1.0/h).norm() - payload.texture->getColor(u, v).norm());
    Eigen::Vector3f ln(-dU, -dV, 1);
    n = (TBN * ln).normalized();

    point += c2 * n * payload.texture->getColor(u, v).norm();  // n提供方向, payload.texture->getColor(u, v).norm()提供大小, c2 是常数, 点沿新的法线方向移动, 造成凹凸

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* components are. Then, accumulate that result on the *result_color* object.
        float r2 = get_euclidean_distance_square(light.position, point);
        Eigen::Vector3f l = (light.position - point).normalized();
        Eigen::Vector3f v = (eye_pos - point).normalized();
        Eigen::Vector3f h = (v + l).normalized();

        Eigen::Vector3f ambient  = ka.array() * amb_light_intensity.array();  // shape = (3,1)
        Eigen::Vector3f diffuse  = kd.array() * (light.intensity / r2).array() * std::max(0.f, normal.dot(l));
        Eigen::Vector3f specular = ks.array() * (light.intensity / r2).array() * std::pow(std::max(0.f, normal.dot(h)), p);

        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;  // 模型的三角形列表, 每个三角形都有对应的法向量和纹理坐标

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    // std::string obj_path = "../models/spot/";
    // std::string obj_path = "D:/x/HF/GAMES101/HF/Assignment3/models/spot/";
    std::string obj_path = "D:/x/GAMES/GAMES101/Assignment3/models/spot/";
    // std::string obj_path = "D:/x/GAMES/GAMES101/Assignment3/models/rick/";

    // Load .obj File
    // bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    bool loadout = Loader.LoadFile(obj_path + "spot_triangulated_good.obj");
    // bool loadout = Loader.LoadFile(obj_path + "Rick(fixed).obj");
    // bool loadout = Loader.LoadFile(obj_path + "Rick.obj");

    // auto texture_path = "Rick_d.png";
    // auto texture_path = "rock.png";
    // auto texture_path = "rock.png";
    auto texture_path = "spot.png";
    
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();  // 纹理有多少三角形就new多少三角形
            for(int j=0;j<3;j++)
            {
                t->setVertex(j, Vector4f(mesh.Vertices[i+j].Position.X, mesh.Vertices[i+j].Position.Y, mesh.Vertices[i+j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i+j].Normal.X, mesh.Vertices[i+j].Normal.Y, mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j, Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    r.set_texture(Texture(obj_path + texture_path));

    // std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = normal_fragment_shader;
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = texture_fragment_shader;
    // std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;
    // std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = bump_fragment_shader;
    // std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = displacement_fragment_shader;


    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    // Eigen::Vector3f eye_pos = {0,2,9};
    Eigen::Vector3f eye_pos = {0,0,10};
    // Eigen::Vector3f eye_pos = {0,0,30};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        clock_t t1 = clock();
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        angle = 0;
        // if (key == 'a' )
        // {
        //     angle -= 10;
        // }
        // else if (key == 'd')
        // {
        //     angle += 10;
        // }

        std::cout<<"Multi Threads: "<<(clock() - t1) * 1.0 / CLOCKS_PER_SEC<< "s"<<std::endl;
        std::cout << "frame count: " << frame_count++ << '\n';
    }
    return 0;
}
