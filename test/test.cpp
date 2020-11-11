#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<eigen3/Eigen/Eigen>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
using namespace Eigen;

static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f Q(x, y, _v[0][2]);  // 要判断的点
    // 顺时针算, 所以都是负的
    return ((_v[0]-_v[1]).cross(Q-_v[1])[2]<0 && (_v[2]-_v[0]).cross(Q-_v[0])[2]<0 && (_v[1]-_v[2]).cross(Q-_v[2])[2]<0)/* || 
           ((_v[0]-_v[1]).cross(Q-_v[1])[2]>0 && (_v[2]-_v[0]).cross(Q-_v[0])[2]>0 && (_v[1]-_v[2]).cross(Q-_v[2])[2]>0)*/;
}

int main()
{
	std::vector<Eigen::Vector3f> pos
            {
                    {3, 0, -2},
                    {0, 3, -3},
                    {-3, 0, -3},
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
    // cout<<(v[0]-v[1]).cross(Q-v[1]);
    cout<<insideTriangle(1,1,v)<<endl;
    cout<<insideTriangle(-1,1,v)<<endl;
    cout<<insideTriangle(0,1,v)<<endl;
    cout<<insideTriangle(0,2,v)<<endl;
}

