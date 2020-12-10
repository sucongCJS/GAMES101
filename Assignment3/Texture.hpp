//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <iostream>

class Texture{
private:
    cv::Mat image_data;

    /*
    * 线性插值
    * x 取值范围[0, 1]
    * */
    float lerp(float x, float v_0, float v_1)
    {
        // return x * v_1 + (1-x) * v_0;
        return v_0 + x * (v_1 - v_0);
    }

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor2(float u, float v)
    {
        // u, v 取值范围[0, 1]
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        // std::cout<<getColorBilinear(u, v)[1]<<'\n';
        // std::cout<<getColorBilinear2(u, v)[1]<<'\n'<<'\n';
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    // Eigen::Vector3f getColor(float u, float v)
    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        int a = floor(u_img);
        int b = floor(v_img);
        int c = ceil(u_img);
        int d = ceil(v_img);
        auto u00 = image_data.at<cv::Vec3b>(a, b);  // RGB
        auto u10 = image_data.at<cv::Vec3b>(c,  b);
        auto u01 = image_data.at<cv::Vec3b>(a, d);
        auto u11 = image_data.at<cv::Vec3b>(c,  d);

        float x_u = u_img - floor(u_img);  // 离得越远, 权重越小
        float x_v = v_img - floor(v_img);
        Eigen::Vector3f return_color;
        for(int i=0; i<3; i++)  // RGB 三个颜色, 所以循环三次
        {
            auto u0 = lerp(x_u, u00[i], u10[i]);
            auto u1 = lerp(x_u, u01[i], u11[i]);
            return_color[i] = lerp(x_v, u0, u1);
        }
        return return_color;
    }

    Eigen::Vector3f getColor(float u, float v)
    // Eigen::Vector3f getColorBilinear2(float u, float v)
    // Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        cv::Mat patch;
        // std::clog << "v_img=" << v_img << " | u_img=" << u_img << std::endl;
        // std::clog << "height=" << height << " | width=" << width << std::endl;        
        cv::getRectSubPix(image_data, cv::Size(1, 1), cv::Point2f(u_img, v_img), patch);
        auto color = patch.at<cv::Vec3b>(0, 0);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }
};
#endif //RASTERIZER_TEXTURE_H
