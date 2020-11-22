#ifndef BARCODE_SHOW
#define BARCODE_SHOW

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include "methods.h"
#include "gradient_map.h"



void draw_markup(IplImage* frame, int width, int height);

void draw_circles(IplImage* frame, int x, int y);

void draw_corner_circles(IplImage* frame, int x, int y);

void draw_corner2_circles(IplImage* frame, int x, int y);

void show_persp_transformed(IplImage* in_img,
         const int L, const int R, const int U, const int D,
         int a1, int b1, int a2, int b2,
         int a3, int b3, int a4, int b4) ;

int count_AT ( cv::Mat img, int x, int y);

int count_Global_AT ( cv::Mat img, int x, int y);

int count_local_AT ( cv::Mat img, int x, int y);

int barcode_show_out(IplImage &frame, const int width, const int height);

void show_BarcodeDetector();

#endif
