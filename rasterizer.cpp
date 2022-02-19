#include "rasterizer.hpp"
#include <vector>
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f>& positions)
{
    int id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i>& indices)
{
    int id = get_next_id();
    ind_buf.emplace(id, indices);
    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f>& colors)
{
    int id = get_next_id();
    col_buf.emplace(id, colors);
    return {id};
}

rst::nor_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    int id = get_next_id();
    nor_buf.emplace(id, normals);
    return {id};
}

void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1)
    {
        if (dx >= 0)
        {
            x = x1;
            y = y1;
            xe = x2;
        }
        else
        {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector3f point(x, y, 1.0f);
        set_pixel(point, line_color);

        for (i = 0; x < xe; ++i)
        {
            x = x + 1;
            if (px < 0)
            {
                px = px + 2 * dy1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    y += 1;
                }
                else
                {
                    y -= 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            Eigen::Vector3f point(x, y, 1.0f);
            set_pixel(point, line_color);
        }
    }
    else
    {
        if (dy >= 0)
        {
            x = x1;
            y = y1;
            ye = y2;
        }
        else
        {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector3f point(x, y, 1.0f);
        set_pixel(point, line_color);
        for (i = 0; y < ye; ++i)
        {
            y += 1;
            if (py < 0)
            {
                py += 2 * dx1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    x += 1;
                }
                else
                {
                    x -= 1;
                }
                py += 2 * (dx1 - dy1);
            }
            Eigen::Vector3f point(x, y, 1.0f);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

void rst::rasterizer::draw(rst::pos_buf_id pos_buffer, rst::ind_buf_id ind_buffer, rst::col_buf_id col_buffer, rst::Primitive type)
{
    if (type != rst::Primitive::Triangle)
    {
        throw std::runtime_error("Drawing primitives other than triangle is not implemented yet!");
    }
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;

    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
            mvp * to_vec4(buf[i[0]]),
            mvp * to_vec4(buf[i[1]]),
            mvp * to_vec4(buf[i[2]])
        };

        // touying -> shikou
        //w control Suofang
        for (auto& vec : v)
        {
            vec /= vec.w();
        }

        for (auto& vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
        }

        for (int j = 0; j < 3; ++j)
        {
            t.setColor(j, col[i[j]].x(), col[i[j]].y(), col[i[j]].z());
        }
        // guanshanhua xiankuangmoxing
        //rasterize_wireframe(t);

        // guanshanhua zhuosemoxing
        rasterize_triangle(t);
    }
}

void rst::rasterizer::draw(std::vector<Triangle*>& TriangleList)
{
    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;

    for (const auto& t: TriangleList)
    {
        Triangle newtri = *t;

        auto v = t->toVector4();

        std::array<Eigen::Vector4f, 3> mm {
                (view * model * v[0]),
                (view * model * v[1]),
                (view * model * v[2])
        };

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v) {
            return v.template head<3>();
        });

        Eigen::Vector4f vv[] = {
            mvp * v[0],
            mvp * v[1],
            mvp * v[2]
        };
        
        // after projection transform, the w is not 1
        for (auto& vec : vv) 
        {
            vec.x()/=vec.w();
            vec.y()/=vec.w();
            vec.z()/=vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
            inv_trans * to_vec4(t->normal[0], 0.0f),
            inv_trans * to_vec4(t->normal[1], 0.0f),
            inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        for (auto& vert : vv)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            newtri.setVertex(i, Vector3f(vv[i].x(), vv[i].y(), vv[i].z()));
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148,121.0,92.0);
        newtri.setColor(1, 148,121.0,92.0);
        newtri.setColor(2, 148,121.0,92.0);

        rasterize_triangle(newtri, viewspace_pos);

    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1, const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f& vert1, const Eigen::Vector2f& vert2, const Eigen::Vector2f& vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

bool insiderTriangle(float x, float y, const Vector3f* _v)
{
    Vector3f point(x, y, 1.0f);
    Vector3f flag[3];

    for (int i = 0; i < 3; ++i)
    {
        Vector3f tri = _v[i] - _v[(i + 1) % 3];
        Vector3f ptri= point - _v[(i + 1) % 3];
        flag[i] = tri.cross(ptri);
    }

    for (int i = 1; i < 3; ++i)
    {
        if (flag[i].z() * flag[i - 1].z() < 0)
        {
            return false;
        }
    }
    return true;
}

static std::tuple<float, float, float> computeBarycentric(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    //float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    float c3 = 1 - c2 - c1;
    return {c1,c2,c3};
}

void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos)
{
    auto v = t.toVector4();
    float xmin = std::min(v[0][0], std::min(v[1][0], v[2][0]));
    float xmax = std::max(v[0][0], std::max(v[1][0], v[2][0]));
    float ymin = std::min(v[0][1], std::min(v[1][1], v[2][1]));
    float ymax = std::max(v[0][1], std::max(v[1][1], v[2][1]));

    int x_min = std::floor(xmin);
    int x_max = std::ceil(xmax);
    int y_min = std::floor(ymin);
    int y_max = std::ceil(ymax);

    bool using_msaa = true;

    for (int i = x_min; i <= x_max; ++i)
    {
        for (int j = y_min; j <= y_max; ++j)
        {
            float x = i + 0.5f;
            float y = j + 0.5f;
            if (!using_msaa && insiderTriangle(x, y, t.v))
            {
                auto[alpha, beta, gamma] = computeBarycentric(x, y, t.v);
                float Z = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                zp *= Z;

                
                if (depth_buf[get_index(i, j)] > zp)
                {
                    
                    auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                    auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                    auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                    auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                    depth_buf[get_index(i, j)] = zp;

                    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;
                    auto pixel_color = fragment_shader(payload);
                    
                    set_pixel(Eigen::Vector3f(i, j, 1), pixel_color);
                }
            } 
            else if (using_msaa)
            {
                int count = msaa_count(i, j, t);
                if (count > 0)
                {
                    auto[alpha, beta, gamma] = computeBarycentric(x, y, t.v);
                    float Z = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    zp *= Z;

                    auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                    auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                    auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                    auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                    depth_buf[get_index(i, j)] = zp;

                    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;
                    auto pixel_color = fragment_shader(payload);

                    // pixel_color = count / 4.0 * pixel_color + frame_buf[get_index(i, j)] * (4 - count) / 4.0;
                    set_pixel(Eigen::Vector3f(i, j, 1), pixel_color);
                }
            }
        }
    }
}


void rst::rasterizer::MSAA(int x, int y, const Triangle& t)
{
    auto v = t.toVector4();
    int count = 0;
    float zmin = std::numeric_limits<float>::infinity();
    Eigen::Vector3f color;
    float d[4][2] = {{0.25, 0.25}, {0.25, 0.75}, {0.75, 0.25}, {0.75, 0.75}};
    for (int i = 0; i < 4; ++i)
    {
        if (insiderTriangle(x + d[i][0], y + d[i][1], t.v))
        {
            
            auto [alpha, beta, gamma] = computeBarycentric(x, y, t.v);
            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;

            if (depth_buf[get_index(x, y)] > z_interpolated)
            {
                count++;
                zmin = std::min(zmin, z_interpolated);
                color = t.get_color(alpha, beta, gamma);
            }
        }
    }
    if (count > 0)
    {
        depth_buf[get_index(x, y)] = zmin;
        set_pixel(Vector3f(x, y, 1), color * (count / 4.f) + (4 - count) * frame_buf[get_index(x, y)] / 4.f);
    }
}

int rst::rasterizer::msaa_count(int x, int y, const Triangle& t)
{
    auto v = t.toVector4();
    int count = 0;

    float d[4][2] = {{0.25, 0.25}, {0.25, 0.75}, {0.75, 0.25}, {0.75, 0.75}};
    for (int i = 0; i < 4; ++i)
    {
        if (insiderTriangle(x + d[i][0], y + d[i][1], t.v))
        {
            
            auto [alpha, beta, gamma] = computeBarycentric(x, y, t.v);
            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;

            if (depth_buf[get_index(x, y)] > z_interpolated)
            {
                count++;
            }
        }
    }
    return count;
}

void rst::rasterizer::rasterize_triangle(const Triangle& t)
{
    auto v = t.toVector4();

    float x_min = std::min({v[0].x(), v[1].x(), v[2].x()});
    float x_max = std::max({v[0].x(), v[1].x(), v[2].x()});
    float y_min = std::min({v[0].y(), v[1].y(), v[2].y()});
    float y_max = std::max({v[0].y(), v[1].y(), v[2].y()});

    int xmin = floor(x_min);
    int xmax = ceil(x_max);
    int ymin = floor(y_min);
    int ymax = ceil(y_max);

    bool using_MSAA = 1;

    for (int x = xmin; x <= x_max; ++x)
    {
        for (int y = ymin; y <= ymax; ++y)
        {
            if (using_MSAA)
            {
                MSAA(x, y, t);
            }
            else
            {
                if (insiderTriangle(x + 0.5f, y + 0.5f, t.v))
                {
                    auto [alpha, beta, gamma] = computeBarycentric(x, y, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    float& depth = depth_buf[get_index(x, y)];
                    if (z_interpolated < depth)
                    {
                        depth = z_interpolated;
                        set_pixel(Vector3f(x, y, z_interpolated), t.get_color(alpha, beta, gamma));
                    }
                }
            }
        }
    }
}

void rst::rasterizer::rasterize_wireframe(const Triangle& t)
{
    draw_line(t.c(), t.a());
    draw_line(t.a(), t.b());
    draw_line(t.b(), t.c());
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{102, 249, 207});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    if (point.x() < 0.0 || point.x() > width ||
        point.y() < 0.0 || point.y() > height) return;
    int ind = (height - point.y()) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}
