#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

#pragma once

class Texture
{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        // auto u_img = u * width;
        // auto v_img = (1 - v) * height;
        // auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        // return Eigen::Vector3f(color[0], color[1], color[2]);

        u=std::min(1.0f, std::max(0.0f, u)); // you should set constraints
        v=std::min(1.0f, std::max(0.0f, v));
        
        auto u_img = u * width; 
        auto v_img = (1 - v) * height;

        auto u_min=std::floor(u_img);
        auto u_max=std::ceil(u_img);
        auto v_min=std::floor(v_img);
        auto v_max=std::ceil(v_img);

        // Color_11 Color_12
        // Color_21 Color_22
        // in opencv, higher y in image->smaller v in texture
        auto color_11=image_data.at<cv::Vec3b>(v_min,u_min);
        auto color_12=image_data.at<cv::Vec3b>(v_min,u_max);
        auto color_21=image_data.at<cv::Vec3b>(v_max,u_min);
        auto color_22=image_data.at<cv::Vec3b>(v_max,u_max);

        auto color1=(u_img-u_min)*color_12+(u_max-u_img)*color_11;
        auto color2=(u_img-u_min)*color_22+(u_max-u_img)*color_21;

        auto color=(v_img-v_min)*color2+(v_max-v_img)*color1;

        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

};