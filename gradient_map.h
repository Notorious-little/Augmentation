#ifndef LOCAL_GRAD
#define LOCAL_GRAD

#include <cassert>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

double* localGradientMap(const cv::Mat &input_img, double* Map,
                         const double GlobalThreshold, const double AV,
                         const double M1, const int detnum);

#endif
