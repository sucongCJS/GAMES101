#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<eigen3/Eigen/Eigen>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
using namespace Eigen;

static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f Q(x, y, 0);  // 要判断的点, 像素的中心是否在三角形内来决定像素是否在三角形内
    // 顺时针算, 所以都是负的
    return ((_v[0]-_v[1]).cross(Q-_v[1]).z()<0 && 
            (_v[2]-_v[0]).cross(Q-_v[0]).z()<0 && 
            (_v[1]-_v[2]).cross(Q-_v[2]).z()<0);
}

static float insideTrianglePercent(int x, int y, const Vector3f* _v)
{
    float percent = 0;
    percent += insideTriangle(x+0.25, y+0.25, _v) * 0.25 + 
               insideTriangle(x+0.75, y+0.25, _v) * 0.25 + 
               insideTriangle(x+0.25, y+0.75, _v) * 0.25 + 
               insideTriangle(x+0.75, y+0.75, _v) * 0.25;
    return percent;
}

int main()
{
	std::vector<Eigen::Vector3f> pos
            {
                    {2, 0.5, 0},
                    {0, 2, 0},
                    {-2, 0.5, 0},
            };

    std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
    pos_buf.emplace(0, pos);
    auto& buf = pos_buf[0];
    Vector3f v[3];
    for(int i=0; i<3; i++)
    {
        v[i] = buf[i];
    }
    Vector3f Q(0, 1, 0);
    // cout<<(v[0]-v[1]).cross(Q-v[1]);R
    // cout<<insideTrianglePercent(1,1,v)<<endl;
    // cout<<insideTrianglePercent(-1,1,v)<<endl;
    // cout<<insideTrianglePercent(0,1,v)<<endl;
    // cout<<insideTrianglePercent(0,2,v)<<endl;
    cout<<insideTrianglePercent(0,0,v);
    // for(float i=-4; i<4; i+=1){
    //     for(float j=-4; j<4; j+=1){
    //         cout<<insideTrianglePercent(i, j, v)<<endl;
    //     }
    // }
}

