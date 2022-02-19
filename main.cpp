#include <iostream>
#include <opencv2/opencv.hpp>

#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

constexpr double MY_PI = 3.1415926;

inline double Degree(double angle) {return angle * MY_PI / 180.0;};

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
    angle = Degree(angle);
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

    Eigen::Matrix4f persp_to_ortho;
    persp_to_ortho << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, zNear * zFar,
        0, 0, -1, 0;
    projection = persp_to_ortho * projection;

    Eigen::Matrix4f T_ortho, S_ortho;
    T_ortho << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, (zNear + zFar) / 2,
        0, 0, 0, 1;

    float t = abs(zNear) * tan(Degree(eye_fov) / 2.0);
    S_ortho << 1 / (t * aspect_ratio), 0, 0, 0,
        0, 1 / t, 0, 0,
        0, 0, -2 / (zFar - zNear), 0, 
        0, 0, 0, 1;
    projection = T_ortho * projection;
    projection = S_ortho * projection;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = 255.0f * (payload.normal + Eigen::Vector3f(1, 1, 1) / 2.0f);
    return return_color;
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
    Eigen::Vector3f texture_color = payload.texture->getColor(payload.tex_coords[0], payload.tex_coords[1]);

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color/ 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        Eigen::Vector3f l = light.position - point;
        auto distance = l.dot(l);

        l = l.normalized();
        Eigen::Vector3f v = eye_pos - point;
        v = v.normalized();
        normal = normal.normalized();

        auto h = (l + v).normalized();
        auto ambinet = ka.cwiseProduct(amb_light_intensity);
        auto diffuse = kd.cwiseProduct(light.intensity / distance) * std::max(0.0f, normal.dot(l));
        auto specular = ks.cwiseProduct(light.intensity / distance) * std::pow(std::max(0.0f, normal.dot(h)), p);

        result_color += (ambinet + diffuse + specular);
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
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

    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        Eigen::Vector3f l = light.position - point;
        auto r2 = l.dot(l);
        l = l.normalized();
        
        Eigen::Vector3f v = eye_pos - point;
        v = v.normalized();
        normal = normal.normalized();

        auto h = (l + v).normalized();

        auto ambinet = ka.cwiseProduct(amb_light_intensity);
        auto diffuse = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.dot(l));
        auto specular = ks.cwiseProduct(light.intensity / r2) * std::pow(std::max(0.0f, normal.dot(h)), p);

        result_color += (ambinet + diffuse + specular);
    }

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
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

    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;

    auto n = normal;
    auto x = n[0], y = n[1], z = n[2];
    float temp = std::sqrt(x * x + z * z);
    Eigen::Vector3f t(x * y / temp, temp, z * y / temp);
    Eigen::Vector3f b = n.cross(t);
    Eigen::Matrix3f TBN;
    TBN << t.x(), b.x(), n.x(),
        t.y(), b.y(), n.y(),
        t.z(), b.z(), b.z();

    float h = payload.texture->height, w = payload.texture->width;
    float u = payload.tex_coords.x(), v = payload.tex_coords.y();
    float dU = kh * kn * (payload.texture->getColor((u + 1 / w), v).norm() - payload.texture->getColor(u, v).norm());
    float dV = kh * kn * (payload.texture->getColor(u, (v + 1 / h)).norm() - payload.texture->getColor(u, v).norm());
    Eigen::Vector3f ln(-dU, -dV, 1);
    point += kn * n * payload.texture->getColor(u, v).norm();

    normal = (TBN * ln).normalized();

    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;

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

    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    auto x=normal.x(), y=normal.y(), z=normal.z();
    Eigen::Vector3f t(x*y/std::sqrt(x*x+z*z), std::sqrt(x*x+z*z), z*y/sqrt(x*x+z*z));
    Eigen::Vector3f b=normal.cross(t);
    Eigen::Matrix3f TBN;
    TBN << t.x(),b.x(),normal.x(),
           t.y(),b.y(),normal.y(),
           t.z(),b.z(),normal.z();

    float h=payload.texture->height, w=payload.texture->width, u=payload.tex_coords.x(), v=payload.tex_coords.y();
    float dU=kh*kn*(payload.texture->getColor(u+1/w,v).norm()-payload.texture->getColor(u,v).norm()); 
    float dV=kh*kn*(payload.texture->getColor(u,v+1/h).norm()-payload.texture->getColor(u,v).norm());
    // Vector ln = (-dU, -dV, 1)
    Eigen::Vector3f ln(-dU, -dV, 1);
    // Position p = p + kn * n * h(u,v)
    point += kn*normal*payload.texture->getColor(u,v).norm();
    // Normal n = normalize(TBN * ln)
    normal = (TBN * ln).normalized();

    Eigen::Vector3f result_color(0, 0, 0);

    for (auto& light : lights)
    {
        Eigen::Vector3f l = light.position - point;
        Eigen::Vector3f v = eye_pos-point; 

        float r2=l.dot(l);

        l = l.normalized();
        v = v.normalized();

        Eigen::Vector3f La=ka.cwiseProduct(amb_light_intensity); 

        Eigen::Vector3f Ld=kd.cwiseProduct(light.intensity / r2)*std::max(0.0f, normal.normalized().dot(l));

        Eigen::Vector3f h=(l+v).normalized();
        Eigen::Vector3f Ls=ks.cwiseProduct(light.intensity / r2)*std::pow(std::max(0.0f, normal.normalized().dot(h)), p);
        
        result_color+=(La+Ld+Ls);

    }

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;

    std::string filename = "output.png";
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc == 2)
    {
        filename = argv[1];
    }
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";
    auto texture_path = "hmap.jpg";

    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for (auto& mesh : Loader.LoadedMeshes)
    {
        for (int i = 0; i < mesh.Vertices.size(); i += 3)
        {
            Triangle* t = new Triangle();
            for (int j = 0; j < 3; ++j)
            {
                t->setVertex(j, Vector3f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y, mesh.Vertices[i + j].Position.Z));
                t->setTexCoord(j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X, mesh.Vertices[i + j].TextureCoordinate.Y));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y, mesh.Vertices[i + j].Normal.Z));
            }
            TriangleList.emplace_back(t);          
        }
    }

    rst::rasterizer r(700, 700);

    r.set_texture(Texture(obj_path + texture_path));

    Vector3f eye_pos(0, 0, 10);
    r.set_fragment_shader(displacement_fragment_shader);

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

