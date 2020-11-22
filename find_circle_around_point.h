#ifndef FIND_CIRCLES
#define FIND_CIRCLES

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "gradient_map.h"

int find_circles(cv::Mat &input_img, int N);

#endif
