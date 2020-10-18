#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "gradient_map.h"

div_t div(int numer, int denom);


cv::Mat draw_GM_contoured_img(const cv::Mat &input_img,
                              const double* Map, const int Map_size);
