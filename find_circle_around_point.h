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

void find_circle_around_point(int x, int y, int R, cv::Mat &input_img);
