#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<eigen3/Eigen/Eigen>
#include<opencv2/opencv.hpp>
#include<iostream>
#include <string>
#include <functional>
#include <optional>
using namespace cv;
using namespace Eigen;
 
// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b) {
    if (b)
        return "Godzilla";
    return {};
}

int main()
{
	std::cout<< __cplusplus/100%100;
}