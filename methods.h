#ifndef METHODS
#define METHODS

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>

cv::Mat autoContrast(const cv::Mat &img, const int quantil,
                      const int L, const int R, const int U, const int D);


void labelText(cv::Mat img, const int Amount);


void slidingNormalMatrixCompute(double* elements, const int MatrixSizeParam);


cv::Mat blurByGaussMatrix(const cv::Mat &input_img, const int blurpower,
                           const int L, const int R, const int U, const int D);


cv::Mat blur_monochrome_ByGauss(const cv::Mat &input_img, const int blurpower,
                           const int L, const int R, const int U, const int D);


IplImage* Lapl_of_Gauss_colored(IplImage* frame, const double Param,
                           const int L, const int R, const int U, const int D);


double exp_rand(void);


cv::Mat gaussNoize( const cv::Mat &input_img, const double noize_range);


cv::Mat salt_paperNoize( const cv::Mat &input_img, const int noize_range);


cv::Mat localContours(const cv::Mat &input_img,
                      const double G_AV, const double AV, const double M,
                      const int L, const int R, const int U, const int D);


cv::Mat medianFilter_8UC1 (const cv::Mat &input_img,
                           const int &L, const int &R, const int &U, const int &D);


cv::Mat binarization(const cv::Mat &input_img,
                     const int L, const int R,
                     const int U, const int D);


cv::Mat horizontal_dilatate (const cv::Mat &input_img,
                          const int L, const int R, const int U, const int D);


cv::Mat horizontal_dilatate_rightside (const cv::Mat &input_img, const int L, const int R,
                              const int U, const int D);


cv::Mat horizontal_dilatate_leftside (const cv::Mat &input_img, const int L, const int R,
                              const int U, const int D);


cv::Mat rightside_dilatate (const cv::Mat &input_img, const int L, const int R,
                           const int U, const int D);


cv::Mat leftside_dilatate (const cv::Mat &input_img, const int L, const int R,
                           const int U, const int D);


cv::Mat classic_dilatate(const cv::Mat &input_img, const int L, const int R,
                             const int U, const int D);


cv::Mat classic_erosion(const cv::Mat &input_img,
                        const int L, const int R, const int U, const int D);


cv::Mat vertical_erosion(const cv::Mat &input_img,
                        const int L, const int R, const int U, const int D);


cv::Mat horizontal_erosion(const cv::Mat &input_img, const int L, const int R,
                  const int U, const int D);

cv::Mat draw_barcode(const cv::Mat &input_img, const int L, const int R,
                     const int U, const int D );

#endif
