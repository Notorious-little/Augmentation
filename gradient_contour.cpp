#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>

// Здесь составим "карту градиентов" изображения
// Предварительно оно обработано фильтрами Blur, Autocontrast
// Будем считать, что в точках "карты" такие обозначения :
// 0 - градиент не прошел проверку на локальный максимум в собственной 3-окрестности
// 1 - градиент вертикален
// 2 - градиент горизонтален
// 3 - градиент параллелен "главной диагонали"
// 4 - градиент параллелен "побочной диагонали"

void localGradientMap(const cv::Mat &input_img, double* Map){

    const int h = input_img.rows;
    const int w = input_img.cols;

    const double GK = 1.41 ;
    const double AV = 1.1 ;
    const double M = 0.9;
    const double M1 = 0.9;
    const int detnum = 2;

    int* Rhor = new int [h*w];         int* Rvert = new int [h*w];
    int* Rmain = new int [h*w];        int* Rsub = new int [h*w];
    int* Ghor = new int [h*w];         int* Gvert = new int [h*w];
    int* Gmain = new int [h*w];        int* Gsub = new int [h*w];
    int* Bhor = new int [h*w];         int* Bvert = new int [h*w];
    int* Bmain = new int [h*w];        int* Bsub = new int [h*w];

    int* Shor = new int [h*w];         int* Svert = new int [h*w];
    int* Smain = new int [h*w];        int* Ssub = new int [h*w];

    for(int z = 0; z < h*w; ++z) {
        Rhor[z] = 0;    Rvert[z] = 0;
        Rmain[z] = 0;   Rsub[z] = 0;
        Gmain[z] = 0;   Gsub[z] = 0;
        Gvert[z] = 0;   Ghor[z] = 0;
        Bmain[z] = 0;   Bsub[z] = 0;
        Bvert[z] = 0;   Bhor[z] = 0;
        Smain[z] = 0;   Shor[z] = 0;
        Svert[z] = 0;   Ssub[z] = 0;
    }

    cv::Mat output_img( h , w , CV_8UC1 );

    for (int i = 0; i < h; ++i ){
        for (int j = 0; j < w; ++j ){
            output_img.at<uchar>(i,j) = 0;
        }
    }

    cv::Mat b = input_img.clone();



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 4; i < h-4; ++i ){
        for (int j = 4; j < w-4; ++j ){

        double s = 0;

            for(int k = j - 1; k <= j+1; ++k){
                s += abs (b.at<cv::Vec3b>(i,k)[0] - b.at<cv::Vec3b>(i-2,k)[0])*GK;
                s += abs (b.at<cv::Vec3b>(i+1,k)[0] - b.at<cv::Vec3b>(i-1,k)[0])*GK;
                s += abs (b.at<cv::Vec3b>(i+2,k)[0] - b.at<cv::Vec3b>(i,k)[0])*GK;
            }

            for(int k = i - 1; k <= i+1; ++k){
                s += abs (b.at<cv::Vec3b>(k,j)[0] - b.at<cv::Vec3b>(k,j-2)[0])*GK;
                s += abs (b.at<cv::Vec3b>(k,j+1)[0] - b.at<cv::Vec3b>(k,j-1)[0])*GK;
                s += abs (b.at<cv::Vec3b>(k,j+2)[0] - b.at<cv::Vec3b>(k,j)[0])*GK;
            }

            for(int k = i - 2; k <= i; ++k){
                s += abs (b.at<cv::Vec3b>(k, j-2)[0] - b.at<cv::Vec3b>(k+2,j)[0]);
                s += abs (b.at<cv::Vec3b>(k,j-1)[0] - b.at<cv::Vec3b>(k+2,j+1)[0]);
                s += abs (b.at<cv::Vec3b>(k,j)[0] - b.at<cv::Vec3b>(k+2,j+2)[0]);
            }

            for(int k = i; k <= i + 2; ++k){
                s += abs (b.at<cv::Vec3b>(k, j-2)[0] - b.at<cv::Vec3b>(k-2,j)[0]);
                s += abs (b.at<cv::Vec3b>(k,j-1)[0] - b.at<cv::Vec3b>(k-2,j+1)[0]);
                s += abs (b.at<cv::Vec3b>(k,j)[0] - b.at<cv::Vec3b>(k-2,j+2)[0]);
            }

            s /= 36;



            double s_vert(0), s_hor(0), s_maindig(0), s_subdig(0), s_max(0);

            for(int k = -1; k <= 1; ++k){
                s_vert += abs(b.at<cv::Vec3b>(i-1,j+k)[0] - b.at<cv::Vec3b>(i+1,j+k)[0]);
            }

            for(int k = -1; k <= 1; ++k){
                s_hor += abs(b.at<cv::Vec3b>(i+k,j-1)[0] - b.at<cv::Vec3b>(i+k,j+1)[0]);
            }

            for(int k = 0; k <= 2; ++k){
                s_maindig += abs(b.at<cv::Vec3b>(i-2+k,j+k)[0] - b.at<cv::Vec3b>(i+k,j-2+1)[0]);
            }

            for(int k = 0; k <= 2; ++k){
                s_subdig += abs( b.at<cv::Vec3b>(i-2+k,j-k)[0] - b.at<cv::Vec3b>(i+k,j+2-k)[0]);
            }



            if ( M*s_vert > s_hor && M*GK*s_vert > s_maindig && M*GK*s_vert > s_subdig){
                s_max = s_vert;
                if (s_max > AV*s*3) {
                      if (GK*s_hor < s_maindig* M1 && GK*s_hor < s_subdig* M1) Rvert[i*w+j]+=1;
                }
            } else if(M*s_hor > s_vert && M*GK*s_hor > s_maindig && M*GK*s_hor > s_subdig){
                       s_max = s_hor;
                       if (s_max > AV*s*3) {
                            if (GK*s_vert < s_maindig* M1 && GK*s_vert < s_subdig* M1) Rhor[i*w+j]+=1;
                       }
            } else if( M*s_maindig > GK*s_hor && M*s_maindig > GK* s_vert && M*s_maindig > s_subdig){
                       s_max = s_maindig;
                       if (s_max > AV*s*3) {
                               if(GK*s_vert* M1 > s_subdig && GK*s_hor* M1 > s_subdig) Rmain[i*w+j]+=1;
                           }
            } else if( M*s_subdig > GK*s_hor && M*s_subdig > GK* s_vert && M*s_subdig > s_maindig) {
                s_max = s_subdig;
                if (s_max > AV*s*3) {
                        if(GK*s_vert* M1 > s_maindig && GK*s_hor* M1 > s_maindig) Rsub[i*w+j]+=1;
                }
              }
        }
    }


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GREEN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 4; i < h-4; ++i ){
        for (int j = 4; j < w-4; ++j ){

        double s = 0;

            for(int k = j - 1; k <= j+1; ++k){
                s += abs (b.at<cv::Vec3b>(i,k)[1] - b.at<cv::Vec3b>(i-2,k)[1])*GK;
                s += abs (b.at<cv::Vec3b>(i+1,k)[1] - b.at<cv::Vec3b>(i-1,k)[1])*GK;
                s += abs (b.at<cv::Vec3b>(i+2,k)[1] - b.at<cv::Vec3b>(i,k)[1])*GK;
            }

            for(int k = i - 1; k <= i+1; ++k){
                s += abs (b.at<cv::Vec3b>(k,j)[1] - b.at<cv::Vec3b>(k,j-2)[1])*GK;
                s += abs (b.at<cv::Vec3b>(k,j+1)[1] - b.at<cv::Vec3b>(k,j-1)[1])*GK;
                s += abs (b.at<cv::Vec3b>(k,j+2)[1] - b.at<cv::Vec3b>(k,j)[1])*GK;
            }

            for(int k = i - 2; k <= i; ++k){
                s += abs (b.at<cv::Vec3b>(k, j-2)[1] - b.at<cv::Vec3b>(k+2,j)[1]);
                s += abs (b.at<cv::Vec3b>(k,j-1)[1] - b.at<cv::Vec3b>(k+2,j+1)[1]);
                s += abs (b.at<cv::Vec3b>(k,j)[1] - b.at<cv::Vec3b>(k+2,j+2)[1]);
            }

            for(int k = i; k <= i + 2; ++k){
                s += abs (b.at<cv::Vec3b>(k, j-2)[1] - b.at<cv::Vec3b>(k-2,j)[1]);
                s += abs (b.at<cv::Vec3b>(k,j-1)[1] - b.at<cv::Vec3b>(k-2,j+1)[1]);
                s += abs (b.at<cv::Vec3b>(k,j)[1] - b.at<cv::Vec3b>(k-2,j+2)[1]);
            }

            s /= 36;



            double s_vert(0), s_hor(0), s_maindig(0), s_subdig(0), s_max(0);

            for(int k = -1; k <= 1; ++k){
                s_vert += abs(b.at<cv::Vec3b>(i-1,j+k)[1] - b.at<cv::Vec3b>(i+1,j+k)[1]);
            }

            for(int k = -1; k <= 1; ++k){
                s_hor += abs(b.at<cv::Vec3b>(i+k,j-1)[1] - b.at<cv::Vec3b>(i+k,j+1)[1]);
            }

            for(int k = 0; k <= 2; ++k){
                s_maindig += abs(b.at<cv::Vec3b>(i-2+k,j+k)[1] - b.at<cv::Vec3b>(i+k,j-2+1)[1]);
            }

            for(int k = 0; k <= 2; ++k){
                s_subdig += abs( b.at<cv::Vec3b>(i-2+k,j-k)[1] - b.at<cv::Vec3b>(i+k,j+2-k)[1]);
            }



            if ( M*s_vert > s_hor && M*GK*s_vert > s_maindig && M*GK*s_vert > s_subdig){
                s_max = s_vert;
                if (s_max > AV*s*3) {
                      if (GK*s_hor < s_maindig* M1 && GK*s_hor < s_subdig* M1) Gvert[i*w+j]+=1;
                }
            } else if(M*s_hor > s_vert && M*GK*s_hor > s_maindig && M*GK*s_hor > s_subdig){
                       s_max = s_hor;
                       if (s_max > AV*s*3) {
                            if (GK*s_vert < s_maindig* M1 && GK*s_vert < s_subdig* M1) Ghor[i*w+j]+=1;
                       }
            } else if( M*s_maindig > GK*s_hor && M*s_maindig > GK* s_vert && M*s_maindig > s_subdig){
                       s_max = s_maindig;
                       if (s_max > AV*s*3) {
                               if(GK*s_vert* M1 > s_subdig && GK*s_hor* M1 > s_subdig) Gmain[i*w+j]+=1;
                           }
            } else if( M*s_subdig > GK*s_hor && M*s_subdig > GK* s_vert && M*s_subdig > s_maindig) {
                s_max = s_subdig;
                if (s_max > AV*s*3) {
                        if(GK*s_vert* M1 > s_maindig && GK*s_hor* M1 > s_maindig) Gsub[i*w+j]+=1;
                }
              }
        }
    }


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~BLUE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 4; i < h-4; ++i ){
        for (int j = 4; j < w-4; ++j ){

        double s = 0;

            for(int k = j - 1; k <= j+1; ++k){
                s += abs (b.at<cv::Vec3b>(i,k)[2] - b.at<cv::Vec3b>(i-2,k)[2])*GK;
                s += abs (b.at<cv::Vec3b>(i+1,k)[2] - b.at<cv::Vec3b>(i-1,k)[2])*GK;
                s += abs (b.at<cv::Vec3b>(i+2,k)[2] - b.at<cv::Vec3b>(i,k)[2])*GK;
            }

            for(int k = i - 1; k <= i+1; ++k){
                s += abs (b.at<cv::Vec3b>(k,j)[2] - b.at<cv::Vec3b>(k,j-2)[2])*GK;
                s += abs (b.at<cv::Vec3b>(k,j+1)[2] - b.at<cv::Vec3b>(k,j-1)[2])*GK;
                s += abs (b.at<cv::Vec3b>(k,j+2)[2] - b.at<cv::Vec3b>(k,j)[2])*GK;
            }

            for(int k = i - 2; k <= i; ++k){
                s += abs (b.at<cv::Vec3b>(k, j-2)[2] - b.at<cv::Vec3b>(k+2,j)[2]);
                s += abs (b.at<cv::Vec3b>(k,j-1)[2] - b.at<cv::Vec3b>(k+2,j+1)[2]);
                s += abs (b.at<cv::Vec3b>(k,j)[2] - b.at<cv::Vec3b>(k+2,j+2)[2]);
            }

            for(int k = i; k <= i + 2; ++k){
                s += abs (b.at<cv::Vec3b>(k, j-2)[2] - b.at<cv::Vec3b>(k-2,j)[2]);
                s += abs (b.at<cv::Vec3b>(k,j-1)[2] - b.at<cv::Vec3b>(k-2,j+1)[2]);
                s += abs (b.at<cv::Vec3b>(k,j)[2] - b.at<cv::Vec3b>(k-2,j+2)[2]);
            }

            s /= 36;



            double s_vert(0), s_hor(0), s_maindig(0), s_subdig(0), s_max(0);

            for(int k = -1; k <= 1; ++k){
                s_vert += abs(b.at<cv::Vec3b>(i-1,j+k)[2] - b.at<cv::Vec3b>(i+1,j+k)[2]);
            }

            for(int k = -1; k <= 1; ++k){
                s_hor += abs(b.at<cv::Vec3b>(i+k,j-1)[2] - b.at<cv::Vec3b>(i+k,j+1)[2]);
            }

            for(int k = 0; k <= 2; ++k){
                s_maindig += abs(b.at<cv::Vec3b>(i-2+k,j+k)[2] - b.at<cv::Vec3b>(i+k,j-2+1)[2]);
            }

            for(int k = 0; k <= 2; ++k){
                s_subdig += abs( b.at<cv::Vec3b>(i-2+k,j-k)[2] - b.at<cv::Vec3b>(i+k,j+2-k)[2]);
            }




            if ( M*s_vert > s_hor && M*GK*s_vert > s_maindig && M*GK*s_vert > s_subdig){
                s_max = s_vert;
                if (s_max > AV*s*3) {
                      if (GK*s_hor < s_maindig* M1 && GK*s_hor < s_subdig* M1) Bvert[i*w+j]+=1;
                }
            } else if(M*s_hor > s_vert && M*GK*s_hor > s_maindig && M*GK*s_hor > s_subdig){
                       s_max = s_hor;
                       if (s_max > AV*s*3) {
                            if (GK*s_vert < s_maindig* M1 && GK*s_vert < s_subdig* M1) Bhor[i*w+j]+=1;
                       }
            } else if( M*s_maindig > GK*s_hor && M*s_maindig > GK* s_vert && M*s_maindig > s_subdig){
                       s_max = s_maindig;
                       if (s_max > AV*s*3) {
                               if(GK*s_vert* M1 > s_subdig && GK*s_hor* M1 > s_subdig) Bmain[i*w+j]+=1;
                           }
            } else if( M*s_subdig > GK*s_hor && M*s_subdig > GK* s_vert && M*s_subdig > s_maindig) {
                s_max = s_subdig;
                if (s_max > AV*s*3) {
                        if(GK*s_vert * M1> s_maindig && GK*s_hor * M1> s_maindig) Bsub[i*w+j]+=1;
                }
              }
        }
    }

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RESULT-SUMM~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for(int z = 0; z < h*w; ++z) {
        Smain[z] += Rmain[z];    Smain[z] += Gmain[z];    Smain[z] += Bmain[z];
        Shor[z] += Rhor[z];    Shor[z] += Ghor[z];    Shor[z] += Bhor[z];
        Svert[z] += Rvert[z];    Svert[z] += Gvert[z];    Svert[z] += Bvert[z];
        Ssub[z] += Rsub[z];    Ssub[z] += Gsub[z];    Ssub[z] += Bsub[z];
    }

    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){

        if (Svert[i*w + j] >= detnum) {
            Map [ i*w + j] = 1;
        } else if(Shor[i*w + j] >= detnum){
            Map [ i*w + j] = 2;
        } else if(Smain[i*w + j] >= detnum){
            Map [ i*w + j] = 3;
        }  else if (Ssub[i*w +j] >= detnum){
            Map [ i*w + j] = 4;
           }
        }
    }

return;
}

