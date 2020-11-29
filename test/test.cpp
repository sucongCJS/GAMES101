#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<eigen3/Eigen/Eigen>
#include<opencv2/opencv.hpp>
#include<iostream>
#include <string>
#include <functional>
#include <optional>
#include <fstream>
using namespace cv;
using namespace Eigen;
using namespace std;

int main()
{
    std::ifstream file("D:/x/HF/GAMES101/HF/Assignment3/models/spotw/spot_triangulated_good.obj");
    cout<<file.is_open();
}