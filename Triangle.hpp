#ifndef RASTERIZER_TRIANGLE_H
#define RASTERIZER_TRIANGLE_H

#include <eigen3/Eigen/Eigen>
#include "Texture.hpp"

using namespace Eigen;

class Triangle
{
  public:
    Vector3f v[3];

    Vector3f color[3];
    Vector2f tex_coords[3];
    Vector3f normal[3];

    Texture* tex = nullptr;
    Triangle();

    Eigen::Vector3f a() const{ return v[0]; }
    Eigen::Vector3f b() const{ return v[1]; }
    Eigen::Vector3f c() const{ return v[2]; }

    void setVertex(int ind, Vector3f ver);
    void setColor(int ind, float r, float g, float b);
    void setNormal(int ind, Vector3f n);
    void setTexCoord(int ind, Vector2f uv);

    void setNormals(const std::array<Vector3f, 3>& normals);
    void setColors(const std::array<Vector3f, 3>& colors);

    Eigen::Vector3f get_color() const { return color[0] * 255; };
    
    Eigen::Vector3f get_color(float a, float b, float c) const
    {
      return color[0] * a * 255.0 + color[1] * b * 255.0 + color[2] * c * 255.0;
    }

    std::array<Vector4f, 3> toVector4() const;
};
#endif