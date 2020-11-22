#ifndef ROI_BARCODE_DETECT
#define ROI_BARCODE_DETECT

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "methods.h"
#include "v2_gradient_map.h"




void compute_trace (const int x1, const int y1,
               const int x3, const int y3, int* horizontal);


int scan_right (const cv::Mat &img, const int L, const int R,
                 const int U, const int D, const int x1, const int y1,
                 const int x3, const int y3, int &x, int &y);


int scan_left (const cv::Mat &img, const int L, const int R,
                 const int U, const int D, const int x1, const int y1,
                 const int x3, const int y3, int &x, int &y);


int detect_min_rectangle( IplImage* frame, cv::Mat &input_img,
                           int Line_L, int Line_R, int Line_U, int Line_D,
                           const int L, const int R, const int U, const int D);

#endif
