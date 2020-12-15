#include<eigen3/Eigen/Eigen>
#include<iostream>
#include <string>
#include <functional>
#include <optional>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace Eigen;
using namespace std;

int width = 1024;
int height = 768;

struct s{
    int a;
    int b;
};

int main(int argc, char** argv)
{
    std::optional<s> o;
    o.emplace();
    o->a = 10;
    cout<<o->a;
}