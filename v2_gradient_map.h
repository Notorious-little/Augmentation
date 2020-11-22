#ifndef CENTRAL_CONNECTED_COMPONENT
#define CENTRAL_CONNECTED_COMPONENT

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>


cv::Mat central_connected_component (const cv::Mat &img,
                int L, int R, int U, int D, int *Square);

void wide_Window_GradientMap(const cv::Mat &input_img, double* Map, const int RGB);

void wide_Window_map(const cv::Mat &input_img, double* Map, int detnum);

void central_connected_component (const double* grad_map, double* comp_map, int h, int w);

#endif


