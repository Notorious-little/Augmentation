#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>

cv::Mat autoContrast(const cv::Mat img, const int quantil);

cv::Mat localContours(const cv::Mat input_img);

void slidingNormalMatrixCompute(double* elements, const int MatrixSizeParam);

cv::Mat blurByGaussMatrix(const cv::Mat input_img, const int blurpower);
