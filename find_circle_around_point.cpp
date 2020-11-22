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
#include "methods.h"
#include "map_filtrations.h"


int find_circles(cv::Mat &input_img, int N){        // N is number of different coins

    int* values = new int[N] {1, 2};

    cv::Mat src = input_img.clone();

   // src = autoContrast(src, 5, 0, input_img.rows, 0, input_img.cols);
    src = blurByGaussMatrix(src, 2,  0, input_img.rows, 0, input_img.cols);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::medianBlur(gray, gray, 3);

    std::vector<cv::Vec3f> circles;

    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                    gray.rows/18,          // Change this value to detect circles with different distances to each other
                    150, 40, 30, 60 );   // Change the last two parameters (min_radius & max_radius) to detect larger circles

       for( size_t i = 0; i < circles.size(); i++ )
       {
           cv::Vec3i c = circles[i];
           cv::Point center = cv::Point(c[0], c[1]);
           cv::circle( src, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
           int radius = c[2];
           cv::circle( src, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
       }


        if ( N > 1){

            int circles_number = circles.size();
            float arr[circles_number];

            for (int i = 0; i < circles_number; ++i){
                arr[i] = circles[i][2];
            }

            for (int i = 0; i < circles_number; ++i){
                for (int j = 0; j < circles_number-1; ++j){
                    if( arr[j+1] < arr[j] ){

                        float tmp = arr[j+1];
                        arr[j+1] = arr[j];
                        arr[j] = tmp;
                    }
                }
            }

            int max_number = 0;
            float max_diff = 0;

            double* diffs = new double[ circles.size() -1 ] {0};

            std::cout<<std::endl;

            for( int i = 0; i <  circles.size() -1; ++i){
                diffs[i] = arr[i+1] - arr[i];
            }

            double* gaps = new double[circles.size() - 1] {0};
            int* gaps_nums = new int[circles.size() - 1] {0};

            for (int i = 0; i < circles.size() - 1; ++i){
                gaps[i] = diffs[i];
                gaps_nums[i] = i;
            }

            for (int i = 0; i < circles.size() - 1; ++i){
                for (int j = 0; j < circles_number-1; ++j){

                    if( gaps[j+1] > gaps[j] ){

                      float tmp = gaps[j+1];
                      gaps[j+1] = gaps[j];
                      gaps[j] = tmp;

                      int num_tmp = gaps_nums[j+1];
                      gaps_nums[j+1] = gaps_nums[j];
                      gaps_nums[j] = num_tmp;
                    }
                }
            }

            for (int i = 0; i < N-1; ++i){
                for (int j = 0; j < N-1; ++j){

                    if( gaps_nums[j+1] < gaps_nums[j] ){

                      int num_tmp = gaps_nums[j+1];
                      gaps_nums[j+1] = gaps_nums[j];
                      gaps_nums[j] = num_tmp;

                    }
                }
            }


            int Amount = values[0] * (circles.size() -1 - gaps_nums[0]);
            Amount += (gaps_nums[N-2]+1) * values[N-1];

            for (int i = 1; i < N-1; ++i){
                Amount += values[i] * abs(gaps_nums[i] - gaps_nums[i-1]);
            }

            std::cout<<" AMOUNT : "<< Amount<<std::endl;

            labelText(src, Amount);

            cv::imshow("Detected Circles", src);
            cv::waitKey(0);

        } else {
           std::cout<< " Incorrect N"<<std::endl;
       }

cv::waitKey(0);

return 0;
}
