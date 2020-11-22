#include <cassert>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "gradient_map.h"


div_t div(int numer, int denom);


cv::Mat draw_GM_contoured_img(const cv::Mat &input_img,
                              const double* Map, const int Map_size){

    const int h = input_img.rows;
    const int w = input_img.cols;

    cv::Mat output_img( h , w , CV_8UC1 );

    for (int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            output_img.at<uchar>( i, j ) = 0;
        }
    }


    for (int i = 0; i < h*w +1; ++i){
        if(Map[i] != 0){
             div_t a = std::div(i,w);
             output_img.at<uchar>( a.quot , i % w) = 255;
         }
    }

return output_img;
}




void build_up_map(double* Map, const int h, const int w){

    for ( int i = 1; i < h-1; ++i){
        for ( int j = 1; j < w-1 ; ++j){

            if (( Map [(i-1)*w + (j-1)] == 3) && (Map [(i+1)*w + (j+1)] == 3)){
                Map[i*w + j] = 3;
            }

            if (( Map [(i+1)*w + (j-1)] == 4) && (Map [(i-1)*w + (j+1)] == 4)){
                Map[i*w + j] = 4;
            }

            if (( Map [(i)*w + (j-1)] == 2) && (Map [(i)*w + (j+1)] == 2)){
                Map[i*w + j] = 2;
            }

            if (( Map [(i-1)*w + (j)] == 1) && (Map [(i+1)*w + (j)] == 1)){
                Map[i*w + j] = 1;
            }


        }
    }
return;
}
