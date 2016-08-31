#pragma once

#include <opencv2\opencv.hpp>
#include <vector>

using namespace cv;

class Pencil sealed {
private:
	struct CK {
		static const int rotationCount = 8;
		static const int ckRange = 10;
		static Mat1f convolutionKernels[rotationCount];

		CK();
	};

	static CK ck; // init convolution kernels

	static void CutMatrix(const Mat1f &mat, OutputArray result);
	static void Smooth(float *hist);
	static void Normalize(float *hist);
	static void ToIntegral(float *hist);
	static void CreateGrayMap(float *srcHist, float *dstHist, uchar *outputGrayMap);

	static float P1(float x);
	static float P2(float x);
	static float P3(float x);
	static float P(float x);

private:
	Mat src;
	Mat sobelMat;
	Mat1f responseImages[CK::rotationCount];

	inline Pencil(InputArray src) : src(src.getMat()) { }

	void Step1();
	void Step2();
	void Step3();
	void Step4();
	void ColorMap(OutputArray result);

public:
	// 彩色的铅笔画
	static void ColorDraw(InputArray src, OutputArray dst);
	// 只有线条的铅笔画
	static void Draw(InputArray src, OutputArray dst);
};