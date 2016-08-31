// OpenCVTest.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "Pencil.h"

using namespace cv;

int main() {
	Mat image = imread(R"(Z:\4752.png)");
	Mat result;
	Pencil::ColorDraw(image, result);
	imshow("pencil drawing", result);
	// imwrite(R"(Z:\test-0.png)", result);
	waitKey();
	return 0;
}

