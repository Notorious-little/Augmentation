#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>

cv::Mat autoContrast(const cv::Mat &img, const int quantil);

void slidingNormalMatrixCompute(double* elements, const int MatrixSizeParam);

cv::Mat blurByGaussMatrix(const cv::Mat &input_img, const int blurpower);

double exp_rand(void);

cv::Mat gaussNoize( const cv::Mat &input_img, const int noize_range);

cv::Mat salt_paperNoize( const cv::Mat &input_img, const int noize_range);

cv::Mat localContours(const cv::Mat &input_img);

cv::Mat medianFilter_8UC1 (const cv::Mat &input_img);
