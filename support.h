#ifndef SUPPORT
#define SUPPORT

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

struct TrackBarPar;

void drawTarget(IplImage* img, int x, int y, int radius);

void drawTarget_2(IplImage* img, int x, int y, int radius);

void myMouseCallback_IMAGE( int event, int x, int y, int flags, void* param );

static void on_GMC_trackbar( int, void* );

void show_LocalContoured();

void show_CannyContoured();

static void on_noized_trackbar( int, void* );

void show_NoizedImages();

void show_BinaryContouredImage( const int GlobalTreshold);

static void on_autoc_trackbar( int, void* );

void show_AutocontrImage();

static void on_blur_trackbar( int, void* );

void show_BluredImage();

void show_WideWindowContoured();

void show_OriginalImage(cv::Mat &img);

#endif


