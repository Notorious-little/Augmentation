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
#include "map_edje_filtration.h"


int find_circles(cv::Mat &input_img, int N){        // N is number of different coins

    cv::Mat src = input_img.clone();
    src = autoContrast(src, 5);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);

    std::vector<cv::Vec3f> circles;

    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                    gray.rows/3,          // change this value to detect circles with different distances to each other
                    100, 30, 40, 110 );   // change the last two parameters (min_radius & max_radius) to detect larger circles

       for( size_t i = 0; i < circles.size(); i++ )
       {
           cv::Vec3i c = circles[i];
           cv::Point center = cv::Point(c[0], c[1]);
           // circle center
           cv::circle( src, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
           // circle outline
           int radius = c[2];
           cv::circle( src, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
       }
       cv::imshow("detected circles", src);

       if ( N > 0){

          int circles_number = circles.size();
          float arr[circles_number];

          for (int i = 0; i < circles_number; ++i){
              arr[i] = circles[i][2];
          }

          for (int i = 0; i < circles_number-1; ++i){
              for (int j = 0; j < circles_number-1; ++j){
                  if( arr[j+1] < arr[j] ){
                      float tmp = arr[j+1];
                      arr[j+1] = arr[j];
                      arr[j] = tmp;
                  }
              }
          }

          for (int i = 0; i < circles_number; ++i){

              std::cout<< arr[i]<< " ";
          }
       }
cv::waitKey(0);

return 0;
}
