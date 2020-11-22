#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "methods.h"
#include "barcode_in_zone.h"
#include "barcode.h"


//scr 3:25 !

void compute_trace (const int x1, const int y1,
               const int x3, const int y3, int* horizontal){

    double tg = abs((double)(y3 - y1)) / abs ((double)(x3 - x1));

    for (int i = 0; i < abs(x3 - x1); ++i){
        horizontal[i] = (int)((double)tg * (double)i );
    }
}




int scan_right (const cv::Mat &img, const int L, const int R,
                 const int U, const int D, const int x1, const int y1,
                 const int x3, const int y3, int &x, int &y){

    if (abs(x3 - x1) < abs(y3 - y1)){

        return -1;
    }

    int* horizontal = new int[abs(x3 - x1)];

    compute_trace (x1, y1, x3, y3, horizontal);

    int summ = 0;
    int tmp_x = 0, tmp_y = 0;

    if (x3 - x1 > 200){

        for( int i = 0; i < abs(x3 - x1); ++i){

            summ = 0;

            for (int h = x1; h < x3; ++h){

                if ((h+i < R) && (img.at<uchar>( h+i, (horizontal[h-x1] + y1) ) == 0)){
                    ++summ;
                    tmp_x = (horizontal[h-x1] + y1);    //  ~|~
                    tmp_y = h+i;
                }
            }

            if (summ <= 5){
                x = tmp_x;
                y = tmp_y;
                return 1;
            }
        }

    }  else if (x1 - x3 > 200){

        for( int i = 0; i < abs(x3 - x1); ++i){

            summ = 0;

            for (int h = x3; h < x1; ++h){

                if ((h-i > L) && (img.at<uchar>(  (y3 - horizontal[h-x3] ), h-i ) == 0)){
                    ++summ;
                    tmp_y = (y3 - horizontal[h-x3] );    //  not viewed
                    tmp_x = h-i;
                }
            }

            if ((summ <= 5) && (summ > 0)){
                x = tmp_x;
                y = tmp_y;
                return 1;
            }
        }
    }

    if ( x != 0 && y != 0){
        return 1;
    } else {
        return -2;
    }
}




int scan_left (const cv::Mat &img, const int L, const int R,
                 const int U, const int D, const int x1, const int y1,
                 const int x3, const int y3, int &x, int &y){

    if (abs(x3 - x1) < abs(y3 - y1)){

        return -1;
    }

    int* horizontal = new int[abs(x3 - x1)];

    compute_trace (x1, y1, x3, y3, horizontal);

    int summ = 0;
    int tmp_x = 0, tmp_y = 0;

    if (x3 - x1 > 200){

        for( int i = 0; i < abs(x3 - x1); ++i){

            summ = 0;

            for (int h = x1; h < x3; ++h){

                if ((h+i > L) && (img.at<uchar>( (horizontal[h-x1] + y1), h+i ) == 0)){  // ~|~
                    ++summ;
                    tmp_x = h+i;
                    tmp_y = (horizontal[h-x1] + y1);
                }
            }

            if ((summ < 5) && ( summ > 0) ){
                x = tmp_x;
                y = tmp_y;
                return 1;
            }
        }

    }  else if (x1 - x3 > 200){


        for( int i = 0; i < abs(x3 - x1); ++i){

            summ = 0;

            for (int h = x3; h < x1; ++h){

                if ((h+i < R) && (img.at<uchar>(  (y3 - horizontal[h-x3] ), h+i ) == 0)){
                    ++summ;
                    tmp_y = (y3 - horizontal[h-x3] );    //  ~|~
                    tmp_x = h+i;
                }
            }

            if ((summ <= 5) && (summ > 0)){
                x = tmp_x;
                y = tmp_y;
                return 1;
            }
        }
    }

    if ( x != 0 && y != 0){
        return 1;
    } else {
        return -2;
    }
}




int detect_min_rectangle( IplImage* frame, cv::Mat &input_img,
                           int Line_L, int Line_R, int Line_U, int Line_D,
                           const int L, const int R, const int U, const int D){

    bool C4 = false;
    int count = U + 10;
    int vert = 0;
    int vert1 = R;

    while (Line_U == 0){

        for ( int i = L+1; i < R-1; ++i){

            if (input_img.at<uchar>(count, i) == 0){

                Line_U = count;
                vert = i;
            }

         }

        ++count;
    }

    if ((Line_U > U-5) && (Line_U < (U+D)/2 )){

      //  draw_circles(frame, vert, Line_U);

    } else {
        return 0;
    }

    count = D-10;

    while (Line_D == 0){

        for ( int i = L+1; i < R-1; ++i){

            if ( input_img.at<uchar>(count, i) == 0) {

                Line_D = count;
                vert1 = i;
            }
         }

        --count;
    }

    if ((Line_D < D-5) && (Line_D > U)) {

      //  draw_circles(frame, vert1, Line_D);

    } else {
        return 0;
    }

/* Here we have 2 points : upper corner (vert, Line_U) and lower corner (vert1, Line_D).
 * Have to find border-lines and their intercections :
 * they are corner points "(P1,P2), (G1, G2)" too */

    int x1{vert}, x2{0}, x3{vert1}, x4{0},
        y1{Line_U}, y2{0}, y3{Line_D}, y4{0};

    if ( (vert > R - 20) || ( vert < L + 20) ||
         (vert1 > R - 20) || ( vert < L + 20)){

        return 0;
    }

    double P1{0}, P2{0}, G1{0}, G2{0};

    // Here : if (vert == vert1){} should be done
    if (abs(vert - vert1) < 180){
        return 0;
    }


    if (vert < vert1){

        int x{0}, y{0};
        int x_opposite{0}, y_opposite{0};

        int scaned = scan_right (input_img, L, R, U, D,
                                 x1, y1, x3, y3, x, y);

        int scaned_opposite = scan_left  (input_img, L, R, U, D,
                                          x1, y1, x3, y3, x_opposite, y_opposite);

        if ((scaned == 1) && (scaned_opposite == 1) &&
            (L + 15 < x) && (x + 15 < R) && (U + 15 < y) && (y + 15< D) &&
            (L + 15 < x_opposite) && (x_opposite + 15 < R) &&
            (U + 15 < y_opposite) && (y_opposite + 15 < D) ){

            draw_circles(frame, vert1, Line_D);
            draw_circles(frame, vert, Line_U);
            draw_corner_circles(frame, x, y);
            draw_corner2_circles(frame, x_opposite, y_opposite);

            show_persp_transformed(frame, L, R, U, D,
                     vert, Line_U, x, y, vert1, Line_D, x_opposite, y_opposite);

            return 1;

        } else {

            return -1;
        }


    } else if (vert1 < vert){

        int x{0}, y{0};
        int x_opposite{0}, y_opposite{0};

        int scaned = scan_right (input_img, L, R, U, D,
                                 x1, y1, x3, y3, x, y);

        int scaned_opposite = scan_left  (input_img, L, R, U, D,
                                          x1, y1, x3, y3, x_opposite, y_opposite);

        if ((scaned == 1) && (scaned_opposite == 1) &&
                (L + 15 < x) && (x + 15 < R) && (U + 15 < y) && (y + 15< D) &&
                (L + 15 < x_opposite) && (x_opposite + 15 < R) &&
                (U + 15 < y_opposite) && (y_opposite + 15 < D) ){

            draw_circles(frame, vert1, Line_D);
            draw_circles(frame, vert, Line_U);
            draw_corner_circles(frame, x, y);
            draw_corner2_circles(frame, x_opposite, y_opposite);

            show_persp_transformed(frame, L, R, U, D,
                       vert1, Line_D,  x , y, vert, Line_U ,x_opposite, y_opposite);


        } else {
            return -1;
        }

    }


}




