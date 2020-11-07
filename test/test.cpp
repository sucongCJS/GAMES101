#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<eigen3/Eigen/Eigen>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
using namespace Eigen;


int main()
{
	Eigen::MatrixXf B(2,0);
	cout<<"a"<<endl;
	cout<<B<<endl;
	cout<<"a"<<endl;
	Eigen::VectorXf b2(2);
	cout<<"a"<<endl;
	cout<<b2<<endl;
	cout<<"a"<<endl;
	cout<<b2.transpose()*B<<endl;
	cout<<"a"<<endl;
}

