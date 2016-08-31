#include "Pencil.h"

// 修剪矩阵，只保留和非零的子矩阵
void Pencil::CutMatrix(const Mat1f &mat, OutputArray result) {
	int top = 0, left = 0;

	for (; left < mat.cols && sum(mat.col(left)).val[0] <= 1. / 256; left++);
	for (; top < mat.rows && sum(mat.row(top)).val[0] <= 1. / 256; top++);

	Mat1f { mat, Range { top, mat.rows - top }, Range { left, mat.cols - left } }.copyTo(result);
}

// 直方图平滑处理
void Pencil::Smooth(float *hist) {
	float src[256];
	memcpy(src, hist, 256 * sizeof(float));

	hist[0] = (src[0] + src[1]) / 2;
	for (int i = 1; i < 255; i++) {
		hist[i] = (src[i - 1] + src[i] + src[i + 1]) / 3;
	}
	hist[255] = (src[254] + src[255]) / 2;
}

// 直方图归一化处理
void Pencil::Normalize(float *hist) {
	float sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += hist[i];
	}

	for (int i = 0; i < 256; i++) {
		hist[i] /= sum;
	}
}

// 概率密度直方图转积分直方图
void Pencil::ToIntegral(float *hist) {
	float sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += hist[i];
		hist[i] = sum;
	}
}

// 根据源直方图和目标直方图，得到灰度映射表
void Pencil::CreateGrayMap(float *srcHist, float *dstHist, uchar *outputGrayMap) {
	Smooth(srcHist);
	Smooth(dstHist);
	Normalize(srcHist);
	Normalize(dstHist);
	ToIntegral(srcHist);
	ToIntegral(dstHist);

	int i = 0, j = 0;
	while (i < 256 && j < 256) {
		if (srcHist[i] <= dstHist[j]) {
			while (i < 256 && srcHist[i] <= dstHist[j]) {
				outputGrayMap[i] = j;
				i++;
			}
		} else {
			while (j < 256 && srcHist[i] > dstHist[j]) j++;
			outputGrayMap[i] = j;
		}
	}
	for (; i < 256; i++) {
		outputGrayMap[i] = 255;
	}

	Mat1b temp { 1, 256, outputGrayMap };
	GaussianBlur(temp, temp, { 7, 1 }, 10);
	memcpy(outputGrayMap, temp.data, 256 * sizeof(uchar));
}

inline float Pencil::P1(float x) {
	const float Sigma = 9;
	return (float) (1 / Sigma * exp((x - 255) / Sigma));
}

inline float Pencil::P2(float x) {
	const float A = 105;
	const float B = 225;
	return A <= x && x <= B ? 1 / (B - A) : 0;
}

inline float Pencil::P3(float x) {
	const float Sigma = 11;
	const float D = 80;
	return 1 / (sqrt(2 * 3.1415926535897931 * Sigma)) * exp(-(x - D) * (x - D) / (2 * Sigma * Sigma));
}

inline float Pencil::P(float x) {
	const float W1 = 0.52f;
	const float W2 = 0.37f;
	const float W3 = 0.11f;
	return W1 * P1(x) + W2 * P2(x) + W3 * P3(x);
}

// 步骤一：边缘检测
void Pencil::Step1() {
	cvtColor(src, sobelMat, COLOR_BGR2GRAY);
	sobelMat.convertTo(sobelMat, CV_32F, 1. / 255);

	Mat gx, gy;
	Sobel(sobelMat, gx, sobelMat.depth(), 1, 0);
	Sobel(sobelMat, gy, sobelMat.depth(), 0, 1);
	absdiff(gx, Scalar::all(0), gx);
	absdiff(gy, Scalar::all(0), gy);

	addWeighted(gx, 0.5, gy, 0.5, 0, sobelMat);
}

// 步骤二：进行8方向的运动模糊，即卷积运算
void Pencil::Step2() {
	for (int i = 0; i < CK::rotationCount; i++) {
		filter2D(sobelMat, responseImages[i], CV_32F, CK::convolutionKernels[i]);
	}
}

// 步骤三：对每个响应图的对应像素，值最大的响应图把该像素设置成sobelMat对应的像素值，其他响应图把该像素设为0
void Pencil::Step3() {
	float *p = (float *) sobelMat.data;
	float *p2[CK::rotationCount];
	for (int i = 0; i < CK::rotationCount; i++) {
		p2[i] = (float *) responseImages[i].data;
	}

	int length = (int) (sobelMat.cols * sobelMat.rows);
	for (int i = 0; i < length; i++) {
		int index = -1;
		float max = 0;
		for (int j = 0; j < CK::rotationCount; j++) {
			if (max < p2[j][i]) {
				max = p2[j][i];
				index = j;
			}
		}

		for (int j = 0; j < CK::rotationCount; j++) {
			p2[j][i] = 0;
		}
		if (index >= 0) p2[index][i] = p[i];
	}
}

// 步骤四：分别对响应图进行对应方向的运动模糊，然后把它们叠加
void Pencil::Step4() {
	sobelMat.setTo(Scalar::all(0));

	for (int i = 0; i < CK::rotationCount; i++) {
		filter2D(responseImages[i], responseImages[i], CV_32F, CK::convolutionKernels[i]);
		add(sobelMat, responseImages[i], sobelMat);
	}

	sobelMat.convertTo(sobelMat, CV_8U, -200, 255);
}

// 修改颜色
void Pencil::ColorMap(OutputArray result) {
	const int lightChannelIndex = 2;
	const bool isHls = 0;
	const int bgr2hls = isHls ? COLOR_BGR2HLS : COLOR_BGR2HSV;
	const int hls2bgr = isHls ? COLOR_HLS2BGR : COLOR_HSV2BGR;

	Mat3b hls { src.rows, src.cols };
	Mat1b light { src.rows, src.cols };

	cvtColor(src, hls, bgr2hls);
	mixChannels(hls, light, { lightChannelIndex, 0 });

	Mat1f srcHist;
	calcHist(std::vector<Mat1b>{ light }, { 0 }, noArray(), srcHist, { 256 }, { 0, 255 });

	float dstHist[256];
	for (int i = 0; i < 256; i++) {
		dstHist[i] = P(i);
	}

	uchar grayMap[256];
	CreateGrayMap((float *) srcHist.data, dstHist, grayMap);

	uchar *p = light.data;
	int length = (int) (light.step * light.rows);
	for (int i = 0; i < length; i++) {
		p[i] = grayMap[p[i]];
	}

	mixChannels(light, hls, { 0, lightChannelIndex });
	cvtColor(hls, hls, hls2bgr);

	const double alpha0 = 0.5;
	Mat3b &ycc = hls;
	Mat1b &gray = light;
	cvtColor(hls, ycc, COLOR_BGR2YCrCb);
	mixChannels(ycc, gray, { 0, 0 });
	addWeighted(gray, 1, sobelMat, alpha0, -255 * alpha0, gray);
	mixChannels(gray, ycc, { 0, 0 });
	cvtColor(ycc, hls, COLOR_YCrCb2BGR);

	const double alpha = 0.3;
	Mat3b &bgr = hls;
	Mat1b &r = light, &g = light, &b = light;
	mixChannels(bgr, r, { 2, 0 });
	addWeighted(r, alpha, sobelMat, 1 - alpha, 0, r);
	mixChannels(r, bgr, { 0, 2 });
	mixChannels(bgr, g, { 1, 0 });
	addWeighted(g, alpha, sobelMat, 1 - alpha, 0, g);
	mixChannels(g, bgr, { 0, 1 });
	mixChannels(bgr, b, { 0, 0 });
	addWeighted(b, alpha, sobelMat, 1 - alpha, 0, b);
	mixChannels(b, bgr, { 0, 0 });

	bgr.copyTo(result);
}

void Pencil::ColorDraw(InputArray src, OutputArray dst) {
	Pencil p = { src };
	p.Step1();
	p.Step2();
	p.Step3();
	p.Step4();
	p.ColorMap(dst);
}

void Pencil::Draw(InputArray src, OutputArray dst) {
	Pencil p = { src };
	p.Step1();
	p.Step2();
	p.Step3();
	p.Step4();
	p.sobelMat.copyTo(dst);
}

Mat1f Pencil::CK::convolutionKernels[Pencil::CK::rotationCount] { };

Pencil::CK Pencil::ck { };

// 初始化8方向卷积核
Pencil::CK::CK() {
	Mat rotationMat = getRotationMatrix2D({ (float) CK::ckRange, (float) CK::ckRange }, 180. / CK::rotationCount, 1);

	Mat1f ck { CK::ckRange * 2 + 1, CK::ckRange * 2 + 1, 0.f };
	float *p = (float *) ck.data + ck.cols * CK::ckRange;
	for (int i = 0; i < CK::ckRange * 2 + 1; i++) {
		p[i] = 1;
	}

	for (int i = 0; i < CK::rotationCount; i++) {
		double s = sum(ck).val[0];
		CutMatrix(ck / s, CK::convolutionKernels[i]);
		warpAffine(ck, ck, rotationMat, { ck.cols, ck.rows }, INTER_CUBIC);
	}
}
