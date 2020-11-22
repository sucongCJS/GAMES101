// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f Q(x, y, 0);  // 要判断的点, 像素的中心是否在三角形内来决定像素是否在三角形内
    // 顺时针算, 所以都是负的
    return ((_v[0]-_v[1]).cross(Q-_v[1]).z()<0 && 
            (_v[2]-_v[0]).cross(Q-_v[0]).z()<0 && 
            (_v[1]-_v[2]).cross(Q-_v[2]).z()<0);
}


// MSAA超采样: 返回像素在三角形内的百分比, 在像素内采四个点, 所以可能的值有0, 0.25, 0.5, 0.75, 1
static float insideTrianglePercent(int x, int y, const Vector3f* _v)
{
    float percent = 0;
    percent += insideTriangle(x+0.25, y+0.25, _v) * 0.25 + 
               insideTriangle(x+0.75, y+0.25, _v) * 0.25 + 
               insideTriangle(x+0.25, y+0.75, _v) * 0.25 + 
               insideTriangle(x+0.75, y+0.75, _v) * 0.25;
    return percent;
}

// 改进, 可以指定采样点数, density必须是能开根号的数
static float insideTrianglePercent(int x, int y, const Vector3f* _v, int density)
{
    float percent = 0;
    float step = sqrt(density);  // 如果density是16, step就是4, 4行每行要取4个点
    float fragment_spacing = 1.0/step;  // 采样点与采样点之间的距离是1/4 = 0.25
    float margin = fragment_spacing/2;  // 邻近边界的采样点和边界的距离是0.25/2 = 0.125
    float weight = 1.0/density;  // 每个采样点的权重, 注意类型

    for(int i=0; i<step; i++)
        for(int j=0; j<step; j++)
            percent += insideTriangle(x + margin+fragment_spacing*i, 
                                      y + margin+fragment_spacing*j, _v) * weight;

    return percent;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float alpha = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float beta =  (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    // float gamma = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    float  gamma = 1-alpha-beta;  // gamma和上面的算法最大有0.000001的不同
    return {alpha, beta, gamma};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];  // 格式为{{0, 1, 2}, {3, 4, 5}}
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)  // i包含一个三角形的三个点的索引 eg.{0, 1, 2}
    {
        Triangle t;
        Eigen::Vector4f v[] = {  // 一个三角形三个点的齐次坐标
                mvp * to_vec4(buf[i[0]], 1.0f),  // 一个三角形的第一个点
                mvp * to_vec4(buf[i[1]], 1.0f),  // 一个三角形的第二个点
                mvp * to_vec4(buf[i[2]], 1.0f)
        };  // mvp所有坐标都在[-1,1]^3范围内

        //Homogeneous division
        for (auto& vec : v) {  // 归一
            vec /= vec.w();
        }

        //Viewport transformation  视口, 根据屏幕大小做适应, 坐标范围也在屏幕大小内
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)  // 用这三个顶点初始化三角形的三个顶点
        {
            t.setVertex(i, v[i].head<3>());  // v[i]是齐次坐标的, 有4个值, t.v是original coordinates的, 只有3个值
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) 
{
    // std::cout << true;
    auto v = t.toVector4();  // 把三个顶点转成齐次坐标形式的
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    // int box_left = t.v[0][0];   int box_right = t.v[0][0];
    // int box_bottom = t.v[0][1]; int box_top = t.v[0][1];
    // for(int i = 1; i < 3; ++i)
    // {
    //     if(t.v[i][0] < box_left) box_left = t.v[i][0];
    //     if(t.v[i][0] > box_right) box_right = t.v[i][0];
    //     if(t.v[i][1] < box_bottom) box_bottom = t.v[i][1];
    //     if(t.v[i][1] > box_top) box_top = t.v[i][1];
    // }
    int box_left = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    int box_right = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    int box_bottom = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    int box_top = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
    for(int x=box_left; x<=box_right; x++)
    {
        for(int y=box_bottom; y<=box_top; y++)
        {
            float percent = insideTrianglePercent(x, y, t.v, 16);  // 4个插值点
            // float percent = insideTriangle(x, y, t.v);  // 不加MSAA抗锯齿
            if(percent > 0)  // 只要像素有部分正三角形内
            {
                // z也是要插值的, 因为三角形的点的z值可能不一样
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;  // 归一化
                
                if(z_interpolated < depth_buf[get_index(x, y)])  // 如果比当前点更靠近相机, 设置像素点颜色并更新深度缓冲区, 越小离得越近
                {
                    set_pixel(x, y, t.getColor(), percent);
                    depth_buf[get_index(x,y)] = z_interpolated;
                }
            }
        }
    }
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
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
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
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(int x, int y, const Eigen::Vector3f& color, float percent)
{
    //old index: auto ind = x + y * width;
    // auto ind = get_index(x, y);
    auto ind = x + y * width;
    frame_buf[ind] = color*percent;
}

// clang-format on