#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>

cv::Mat autoContrast(const cv::Mat img, const int quantil){

    assert(0 <= quantil && quantil <= 100);

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

    int k = (int)(quantil*h*w / 100);

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



cv::Mat localContours(const cv::Mat input_img){

    cv::Mat output_img( input_img.rows , input_img.cols , CV_8UC1 );

    cv::Mat b1 = input_img.clone();

    b1 = autoContrast(b1, 5);

    cvtColor( input_img , b1 , cv::COLOR_BGR2GRAY );
    cv::Mat b;
    b1.convertTo(b, CV_8UC1);

    const int h = input_img.rows;
    const int w = input_img.cols;

    for (int i = 2; i < h-2; ++i ){
        for (int j = 2; j < w-2; ++j ){

        double s = 0;

            for(int k = j - 1; k <= j+1; ++k){
                s += abs (b.at<uchar>(i,k) - b.at<uchar>(i-2,k));
                s += abs (b.at<uchar>(i+1,k) - b.at<uchar>(i-1,k));
                s += abs (b.at<uchar>(i+2,k) - b.at<uchar>(i,k));
            }

            for(int k = i - 1; k <= i+1; ++k){
                s += abs (b.at<uchar>(k,j) - b.at<uchar>(k,j-2));
                s += abs (b.at<uchar>(k,j+1) - b.at<uchar>(k,j-1));
                s += abs (b.at<uchar>(k,j+2) - b.at<uchar>(k,j));
            }

            for(int k = i - 2; k <= i; ++k){
                s += abs (b.at<uchar>(k, j-2) - b.at<uchar>(k+2,j));
                s += abs (b.at<uchar>(k,j-1) - b.at<uchar>(k+2,j+1));
                s += abs (b.at<uchar>(k,j) - b.at<uchar>(k+2,j+2));
            }

            for(int k = i; k <= i + 2; ++k){
                s += abs (b.at<uchar>(k, j-2) - b.at<uchar>(k-2,j));
                s += abs (b.at<uchar>(k,j-1) - b.at<uchar>(k-2,j+1));
                s += abs (b.at<uchar>(k,j) - b.at<uchar>(k-2,j+2));
            }

            s /= 36;

            double s_vert(0), s_hor(0), s_maindig(0), s_subdig(0), s_max(0);

            for(int k = -1; k <= 1; ++k){
                s_vert += abs(b.at<uchar>(i-1,j+k) - b.at<uchar>(i+1,j+k));
            }

            for(int k = -1; k <= 1; ++k){
                s_hor += abs(b.at<uchar>(i+k,j-1) - b.at<uchar>(i+k,j+1));
            }

            for(int k = 0; k <= 2; ++k){
                s_maindig += abs(b.at<uchar>(i-2+k,j+k) - b.at<uchar>(i+k,j-2+1));
            }

            for(int k = 0; k <= 2; ++k){
                s_subdig += abs(b.at<uchar>(i-2-k,j-k) - b.at<uchar>(i-k,j+2-k));
            }



            if ( s_vert >= s_hor && s_vert >= s_maindig && s_vert >= s_subdig){
                s_max = s_vert;
                if (s_max > s*3*1.8) {
                    for(int c =-1 ; c < 1; ++c){
                      if (s_hor < s_maindig && s_hor < s_subdig)  output_img.at<uchar>(i,j+c) = 254;
                    }
                }
            } else if(s_hor >= s_vert && s_hor >= s_maindig && s_hor >= s_subdig){
                       s_max = s_hor;
                       if (s_max > s*3*1.8) {
                           for(int c =-1 ; c < 1; ++c){
                            if (s_vert < s_maindig && s_vert < s_subdig) output_img.at<uchar>(i+c,j) = 254;
                           }
                       }
            } else if(s_maindig >= s_hor && s_maindig >= s_vert && s_maindig >= s_subdig){
                       s_max = s_maindig;
                       if (s_max > s*3*1.6) {
                           for(int c =-1 ; c < 2; ++c){
                               if(s_vert > 1.4*s_subdig && s_hor > 1.4*s_subdig) output_img.at<uchar>(i+c,j+c) = 254;
                           }
                       }
            } else {
                s_max = s_subdig;
                if (s_max > s*3*1.6) {
                    for(int c =-1 ; c < 2; ++c){
                        if(s_vert > 1.4*s_maindig && s_hor > 1.4*s_maindig) output_img.at<uchar>(i+c,j-c) = 254;
                    }
                }
              }
        }
    }

return output_img;
}



void labelText(cv::Mat img){

    int h = img.rows;
    int w = img.cols;

    std::string text = "AUTOCONTRASTED";
    cv::Point textOrg(0, h-7);                       // Местоположение  
    int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;  // Фонт (шрифт)
    double fontScale = 0.5;                          // Размер текста
    cv::Scalar color(200, 100, 50);                  // Цвет

    cv::putText(img, text, textOrg, fontFace, fontScale, color);
}




void slidingNormalMatrixCompute(double* elements, const int MatrixSizeParam){
    assert (MatrixSizeParam >= 1);
							 
    int n = MatrixSizeParam * 2 + 1;
    double disp2 = (double)MatrixSizeParam / 2;
    double div = 0;

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){ 			
            elements[i * n + j] = exp( -((MatrixSizeParam-i)*(MatrixSizeParam-i) + 
                                 (MatrixSizeParam-j)*(MatrixSizeParam-j))/(2*disp2) ) / (2*M_PI*disp2); 
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





cv::Mat blurByGaussMatrix(const cv::Mat input_img, const int blurpower){

    cv::Mat bluring_img = input_img.clone();
    const int n = 2*blurpower+1;
    const int h = input_img.rows;
    const int w = input_img.cols;

    for (int i = blurpower; i < h - blurpower; ++i){
        for (int j = blurpower; j < w - blurpower; ++j){
            bluring_img.at<cv::Vec3b>(i,j)[0] = 0;
            bluring_img.at<cv::Vec3b>(i,j)[1] = 0;
            bluring_img.at<cv::Vec3b>(i,j)[2] = 0;
        }
    }

    double Matrix[ n*n ] = {0};

    slidingNormalMatrixCompute( Matrix , blurpower );

    for (int i = blurpower; i < h - blurpower; ++i){
        for (int j = blurpower; j < w - blurpower; ++j){
            double R=0, G=0, B=0;
            for (int k = i - blurpower; k < i + blurpower + 1; ++k){
                for (int l = j - blurpower; l < j + blurpower +1; ++l){
                    R += ( input_img.at<cv::Vec3b>(k,l)[0] *
                            Matrix[(k - i + blurpower) * n + (l - j + blurpower)]);
                    G += ( input_img.at<cv::Vec3b>(k,l)[1] *
                            Matrix[(k - i + blurpower) * n + (l - j + blurpower)]);
                    B += ( input_img.at<cv::Vec3b>(k,l)[2] *
                            Matrix[(k - i + blurpower) * n + (l - j + blurpower)]);
                }
            }
        bluring_img.at<cv::Vec3b>(i,j)[0] = (int)R;
        bluring_img.at<cv::Vec3b>(i,j)[1] = (int)G;
        bluring_img.at<cv::Vec3b>(i,j)[2] = (int)B;
        }
    }

return bluring_img;
}

