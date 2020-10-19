#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "OO_methods.h"


cv::Mat Autocontrast::makeImage(const cv::Mat &img, Params &p){

    assert(0 <= p.quantil && p.quantil <= 100);

    cv::Mat img_autocontr = img.clone();

    int gistR[256] = {0};
    int gistG[256] = {0};
    int gistB[256] = {0};

    int h = img_autocontr.rows;
    int w = img_autocontr.cols;

    for(int i = 0; i < h; ++i){
        for(int j=0; j<w; ++j){
            ++gistR[ img_autocontr.at<cv::Vec3b>(i, j)[0] ];
            ++gistG[ img_autocontr.at<cv::Vec3b>(i, j)[1] ];
            ++gistB[ img_autocontr.at<cv::Vec3b>(i, j)[2] ];
        }
    }

    int Rmin(0), Rmax(255), Gmin(0),
        Gmax(255), Bmin(0), Bmax(255);               // Мин. и макс. интенсивности цветов старого изобр-я

        while (gistR[Rmin] == 0){
            ++Rmin;
        }

        while (gistR[Rmax] == 0){
            --Rmax;
        }

        while (gistG[Gmin] == 0){
            ++Gmin;
        }

        while (gistG[Gmax] == 0){
            --Gmax;
        }

        while (gistB[Bmin] == 0){
            ++Bmin;
        }

        while (gistB[Bmax] == 0){
            --Bmax;
        }

    int k = (int)(p.quantil * h * w / 100);

    int qRmin(Rmin), qRmax(Rmax), qGmin(Gmin),
        qGmax(Gmax), qBmin(Bmin), qBmax(Bmax);          // quantil-Квантили распр-я для настройки новой интенсивности

    int r(gistR[Rmin]), g(gistG[Gmin]), b(gistB[Bmin]);

    while (r < k){
        ++qRmin;
        r += gistR[qRmin];
    }

    while (g < k){
        ++qGmin;
        g += gistG[qGmin];
    }

    while (b < k){
        ++qBmin;
        b += gistB[qBmin];
    }

    int rM(gistR[Rmax]), gM(gistG[Gmax]), bM(gistB[Bmax]);

    while (rM < k){
        --qRmax;
        rM += gistR[qRmax];
    }

    while (gM < k){
        --qGmax;
        gM += gistG[qGmax];
    }

    while (bM < k){
        --qBmax;
        bM += gistB[qBmax];
    }

    for(int i = 0; i < h; ++i){
        for (int j = 0; j < w; ++j){
            if (img_autocontr.at<cv::Vec3b>(i, j)[0] > qRmin && img_autocontr.at<cv::Vec3b>(i, j)[0] < qRmax )
                img_autocontr.at<cv::Vec3b>(i, j)[0] =
                        (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[0] - qRmin)/(qRmax-qRmin) ) % 256 ;

            if (img_autocontr.at<cv::Vec3b>(i, j)[1] > qGmin && img_autocontr.at<cv::Vec3b>(i, j)[1] < qGmax )
                img_autocontr.at<cv::Vec3b>(i, j)[1] =
                        (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[1] - qGmin)/(qGmax-qGmin) ) % 256 ;

            if (img_autocontr.at<cv::Vec3b>(i, j)[2] > qBmin && img_autocontr.at<cv::Vec3b>(i, j)[2] < qBmax )
                img_autocontr.at<cv::Vec3b>(i, j)[2] =
                        (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[2] - qBmin)/(qBmax-qBmin) ) % 256 ;
        }
    }

return img_autocontr;
}


void Blur::slidingNormalMatrixCompute(const int blurpower, double* elements){
    assert (blurpower >= 1);
							 
    int n = blurpower * 2 + 1;
    double disp2 = (double)(blurpower) / 2;
    double div = 0;

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){ 			
            elements[i * n + j] = exp( -((blurpower-i)*(blurpower-i) +
                                  (blurpower-j)*(blurpower-j))/(2*disp2))
                                  / (2*M_PI*disp2);
            div += elements[i * n + j];
        }
    }
    
    div += 1e-7;                                      // Борьба с будущей погрешностью

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            elements[i * n + j] /= div;
        }
    } 
return;
}





cv::Mat Blur::makeImage(const cv::Mat &input_img, Params &p){

    cv::Mat bluring_img = input_img.clone();

    const int n = 2*p.blurpower+1;
    const int h = input_img.rows;
    const int w = input_img.cols;

    for (int i = p.blurpower; i < h - p.blurpower; ++i){
        for (int j = p.blurpower; j < w - p.blurpower; ++j){
            bluring_img.at<cv::Vec3b>(i,j)[0] = 0;
            bluring_img.at<cv::Vec3b>(i,j)[1] = 0;
            bluring_img.at<cv::Vec3b>(i,j)[2] = 0;
        }
    }

  //  double* elements = new double [(2*p.blurpower+1)*(2*p.blurpower+1)];

    int blurpower = p.blurpower;
    double elements[n*n];

    slidingNormalMatrixCompute( blurpower, elements );

    for (int i = p.blurpower; i < h - p.blurpower; ++i){
        for (int j = p.blurpower; j < w - p.blurpower; ++j){
            double R=0, G=0, B=0;
            for (int k = i - p.blurpower; k < i + p.blurpower + 1; ++k){
                for (int l = j - p.blurpower; l < j + p.blurpower +1; ++l){
                    R += ( input_img.at<cv::Vec3b>(k,l)[0] *
                            elements[(k - i + p.blurpower) * n + (l - j + p.blurpower)]);
                    G += ( input_img.at<cv::Vec3b>(k,l)[1] *
                            elements[(k - i + p.blurpower) * n + (l - j + p.blurpower)]);
                    B += ( input_img.at<cv::Vec3b>(k,l)[2] *
                            elements[(k - i + p.blurpower) * n + (l - j + p.blurpower)]);
                }
            }
        bluring_img.at<cv::Vec3b>(i,j)[0] = (int)R;
        bluring_img.at<cv::Vec3b>(i,j)[1] = (int)G;
        bluring_img.at<cv::Vec3b>(i,j)[2] = (int)B;
        }
    }

  //  delete elements;

return bluring_img;
}


/* #define RAND_MAX 2147483647*/
double GaussNoize::exp_rand(void){

    double a, b;

    a = (double)rand();
    b = (double)rand();

    double s = a*a + b*b;

    if ( abs(s) > 1e-50 ) {
        if ((int)a % 2 == 0){
            return ( a * sqrt(((2) * log(s)) / s) );
        } else {
            return ( (-1)*a * sqrt(((2) * log(s)) / s) );
          }
    } else {
        return 0;
    }
return 0;
}


cv::Mat GaussNoize::makeImage(const cv::Mat &input_img, Params &p){

        cv::Mat noizing_img = input_img.clone();

        int w = noizing_img.cols;
        int h = noizing_img.rows;
        
        for (int i = 0; i < h; ++i){
                for (int j = 0; j < w; ++j){

                    double e = exp_rand();
                    e /= 100;
                    double multiplier = (double)(p.noize_range)/10;

                    noizing_img.at<cv::Vec3b>(i,j)[0] =
                            (int) abs(( ( ( 1 + e*multiplier ) * noizing_img.at<cv::Vec3b>(i,j)[0] ) < 255) ?
                                ( ( 1 + e ) * noizing_img.at<cv::Vec3b>(i,j)[0] ) : 255 )  ;
                    noizing_img.at<cv::Vec3b>(i,j)[1] =
                            (int) abs(( ( ( 1 + e*multiplier ) * noizing_img.at<cv::Vec3b>(i,j)[1] ) < 255) ?
                                ( ( 1 + e ) * noizing_img.at<cv::Vec3b>(i,j)[1] ) : 255 )  ;
                    noizing_img.at<cv::Vec3b>(i,j)[2] =
                            (int) abs(( ( ( 1 + e*multiplier ) * noizing_img.at<cv::Vec3b>(i,j)[2] ) < 255) ?
                                ( ( 1 + e ) * noizing_img.at<cv::Vec3b>(i,j)[2] ) : 255 )  ;
                }
        }

return noizing_img;
}

