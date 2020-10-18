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

// В gradient_map.cpp составили "карту градиентов" Map[h*w] изображения
// Предварительно оно обработано фильтрами Blur, (0-quantil linear) Autocontrast
// Будем считать, что в точках "карты" такие обозначения :
// 0 - градиент не прошел проверку (пороговую) на локальный максимум в собственной 3-окрестности
// 1 - градиент вертикален
// 2 - градиент горизонтален
// 3 - градиент параллелен "главной диагонали"
// 4 - градиент параллелен "побочной диагонали"

div_t div(int numer, int denom);


cv::Mat draw_GM_contoured_img(const cv::Mat &input_img,
                              const double* Map, const int Map_size){

    const int h = input_img.rows;
    const int w = input_img.cols;

    cv::Mat output_img( h , w , CV_8UC1 );

    for (int i = 0; i < h*w; ++i){
        if(Map[i] != 0){
             div_t a = std::div(i,w);
             output_img.at<uchar>( a.quot , i % w) = 255;
         }
    }

return output_img;
}


cv::Mat build_up_map(const cv::Mat &input_img){

    const int h = input_img.rows;
    const int w = input_img.cols;

    cv::Mat output_img( h , w , CV_8UC1 );
}
