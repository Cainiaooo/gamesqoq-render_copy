#include <eigen3/Eigen/Eigen>
#include "Texture.hpp"

#pragma once

struct fragment_shader_payload
{
    Eigen::Vector3f view_pos;
    Eigen::Vector3f color;
    Eigen::Vector3f normal;
    Eigen::Vector2f tex_coords;
    Texture* texture;

    fragment_shader_payload()
    {
        texture = nullptr;
    }

    fragment_shader_payload(const Eigen::Vector3f& col, const Eigen::Vector3f& nor,const Eigen::Vector2f& tc, Texture* tex) :
         color(col), normal(nor), tex_coords(tc), texture(tex) {}
};

struct vertex_shader_payload
{
    Eigen::Vector3f position;
};
