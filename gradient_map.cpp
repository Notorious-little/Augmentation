#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>

// Здесь составим "карту градиентов" Map[h*w] изображения
// Предварительно оно обработано фильтрами Blur, (0-quantil linear) Autocontrast
// Будем считать, что в точках "карты" такие обозначения :
// 0 - градиент не прошел проверку на локальный максимум в собственной 3-окрестности
// 1 - градиент вертикален
// 2 - градиент горизонтален
// 3 - градиент параллелен "главной диагонали"
// 4 - градиент параллелен "побочной диагонали"




void count_local_average(cv::Mat &b, double GK,
                         double &s, int Channel, int i, int j){

    for(int k = j - 1; k <= j+1; ++k){
        s += abs (b.at<cv::Vec3b>(i,k)[Channel] -
                  b.at<cv::Vec3b>(i-2,k)[Channel])*GK;
        s += abs (b.at<cv::Vec3b>(i+1,k)[Channel] -
                  b.at<cv::Vec3b>(i-1,k)[Channel])*GK;
        s += abs (b.at<cv::Vec3b>(i+2,k)[Channel] -
                  b.at<cv::Vec3b>(i,k)[Channel])*GK;
    }

    for(int k = i - 1; k <= i+1; ++k){
        s += abs (b.at<cv::Vec3b>(k,j)[Channel] -
                  b.at<cv::Vec3b>(k,j-2)[Channel])*GK;
        s += abs (b.at<cv::Vec3b>(k,j+1)[Channel] -
                  b.at<cv::Vec3b>(k,j-1)[Channel])*GK;
        s += abs (b.at<cv::Vec3b>(k,j+2)[Channel] -
                  b.at<cv::Vec3b>(k,j)[Channel])*GK;
    }

    for(int k = i - 2; k <= i; ++k){
        s += abs (b.at<cv::Vec3b>(k, j-2)[Channel] -
                  b.at<cv::Vec3b>(k+2,j)[Channel]);
        s += abs (b.at<cv::Vec3b>(k,j-1)[Channel] -
                  b.at<cv::Vec3b>(k+2,j+1)[Channel]);
        s += abs (b.at<cv::Vec3b>(k,j)[Channel] -
                  b.at<cv::Vec3b>(k+2,j+2)[Channel]);
    }

    for(int k = i; k <= i + 2; ++k){
        s += abs (b.at<cv::Vec3b>(k, j-2)[Channel] -
                  b.at<cv::Vec3b>(k-2,j)[Channel]);
        s += abs (b.at<cv::Vec3b>(k,j-1)[Channel] -
                  b.at<cv::Vec3b>(k-2,j+1)[Channel]);
        s += abs (b.at<cv::Vec3b>(k,j)[Channel] -
                  b.at<cv::Vec3b>(k-2,j+2)[Channel]);
    }

    s /= 36;

return;
}



void count_average_diffs (cv::Mat &b, long double &av_diff_vert, long double &av_diff_hor,
                              long double &av_diff_maindig, long double &av_diff_subdig,
                              int h, int w, int Channel){
    av_diff_vert = 0;
    av_diff_hor = 0;
    av_diff_maindig = 0 ;
    av_diff_subdig = 0;

        for (int i = 2; i < h-2; ++i ){

        long double tmp_diff_vert(0), tmp_diff_hor(0),
                    tmp_diff_maindig(0), tmp_diff_subdig(0);

            for (int j = 2; j < w-2; ++j ){

                    tmp_diff_vert += abs(b.at<cv::Vec3b>(i-1,j)[Channel] -
                                         b.at<cv::Vec3b>(i+1,j)[Channel]);
                    tmp_diff_hor += abs(b.at<cv::Vec3b>(i,j-1)[Channel]
                                        - b.at<cv::Vec3b>(i,j+1)[Channel]);

                    tmp_diff_maindig += (abs(b.at<cv::Vec3b>(i-1,-1)[Channel] -
                                             b.at<cv::Vec3b>(i+1,j+1)[Channel])) / sqrt(2);

                    tmp_diff_subdig += (abs(b.at<cv::Vec3b>(i-1,j+1)[Channel] -
                                            b.at<cv::Vec3b>(i+1,j-1)[Channel])) / sqrt(2);

            }

            av_diff_vert    += (tmp_diff_vert / w);
            av_diff_hor     += (tmp_diff_hor / w);
            av_diff_maindig += (tmp_diff_maindig / w);
            av_diff_subdig  += (tmp_diff_subdig / w);
    }

    av_diff_vert    /= h;
    av_diff_hor     /= h;
    av_diff_maindig /= h;
    av_diff_subdig  /= h;

return;
}




void localGradientMap(const cv::Mat &input_img, double* Map,
                      const double G_AV, const double AV,
                      const double M1, const int detnum){

    if (( abs(G_AV) < 0.5) || ( abs(AV) < 0.3) ){
        return;
    }


    const int h = input_img.rows;
    const int w = input_img.cols;

    const double GK = 1.41 ;
    const double M = 1.0;


    int* Rhor = new int [(h+1)*(w+1)] {0};         int* Rvert = new int [(h+1)*(w+1)] {0};
    int* Rmain = new int [(h+1)*(w+1)] {0};       int* Rsub = new int [(h+1)*(w+1)] {0};
    int* Ghor = new int [(h+1)*(w+1)] {0};         int* Gvert = new int [(h+1)*(w+1)] {0};
    int* Gmain = new int [(h+1)*(w+1)] {0};        int* Gsub = new int [(h+1)*(w+1)] {0};
    int* Bhor = new int [(h+1)*(w+1)] {0};        int* Bvert = new int [(h+1)*(w+1)] {0};
    int* Bmain = new int [(h+1)*(w+1)] {0};        int* Bsub = new int [(h+1)*(w+1)] {0};

    int* Shor = new int [(h+1)*(w+1)] {0};         int* Svert = new int [(h+1)*(w+1)] {0};
    int* Smain = new int [(h+1)*(w+1)] {0};        int* Ssub = new int [(h+1)*(w+1)] {0};

    cv::Mat b = input_img.clone();



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    long double av_diff_vert(0), av_diff_hor(0), av_diff_maindig(0), av_diff_subdig(0);

    count_average_diffs (b, av_diff_vert, av_diff_hor, av_diff_maindig, av_diff_subdig, h, w, 0);

    for (int i = 4; i < h-4; ++i ){
        for (int j = 4; j < w-4; ++j ){

            double s = 0;

            count_local_average(b, GK, s, 0, i, j);


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
                if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_vert)) {
                      if (GK*s_hor < s_maindig* M1 && GK*s_hor < s_subdig* M1) Rvert[i*w+j]+=1;
                }
            } else if(M*s_hor > s_vert && M*GK*s_hor > s_maindig && M*GK*s_hor > s_subdig){
                       s_max = s_hor;
                       if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_hor)){
                            if (GK*s_vert < s_maindig* M1 && GK*s_vert < s_subdig* M1) Rhor[i*w+j]+=1;
                       }
            } else if( M*s_maindig > GK*s_hor && M*s_maindig > GK* s_vert && M*s_maindig > s_subdig){
                       s_max = s_maindig;
                       if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_maindig)) {
                               if(GK*s_vert* M1 > s_subdig && GK*s_hor* M1 > s_subdig) Rmain[i*w+j]+=1;
                           }
            } else if( M*s_subdig > GK*s_hor && M*s_subdig > GK* s_vert && M*s_subdig > s_maindig) {
                s_max = s_subdig;
                if ((s_max > AV*s*3)  && (s_max > G_AV * av_diff_subdig) ) {
                        if(GK*s_vert* M1 > s_maindig && GK*s_hor* M1 > s_maindig) Rsub[i*w+j]+=1;
                }
              }
        }
    }


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GREEN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    count_average_diffs (b, av_diff_vert, av_diff_hor, av_diff_maindig, av_diff_subdig, h, w, 1);


    for (int i = 4; i < h-4; ++i ){
        for (int j = 4; j < w-4; ++j ){

        double s = 0;

        count_local_average(b, GK, s, 1, i, j);

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
                if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_vert)) {
                      if (GK*s_hor < s_maindig* M1 && GK*s_hor < s_subdig* M1) Gvert[i*w+j]+=1;
                }
            } else if(M*s_hor > s_vert && M*GK*s_hor > s_maindig && M*GK*s_hor > s_subdig){
                       s_max = s_hor;
                       if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_hor)){
                            if (GK*s_vert < s_maindig* M1 && GK*s_vert < s_subdig* M1) Ghor[i*w+j]+=1;
                       }
            } else if( M*s_maindig > GK*s_hor && M*s_maindig > GK* s_vert && M*s_maindig > s_subdig){
                       s_max = s_maindig;
                       if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_maindig)) {
                               if(GK*s_vert* M1 > s_subdig && GK*s_hor* M1 > s_subdig) Gmain[i*w+j]+=1;
                           }
            } else if( M*s_subdig > GK*s_hor && M*s_subdig > GK* s_vert && M*s_subdig > s_maindig) {
                s_max = s_subdig;
                if ((s_max > AV*s*3)  && (s_max > G_AV * av_diff_subdig) ) {
                        if(GK*s_vert* M1 > s_maindig && GK*s_hor* M1 > s_maindig) Gsub[i*w+j]+=1;
                }
              }
        }
    }


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~BLUE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    count_average_diffs (b, av_diff_vert, av_diff_hor, av_diff_maindig, av_diff_subdig, h, w, 1);

    for (int i = 4; i < h-4; ++i ){
        for (int j = 4; j < w-4; ++j ){

        double s = 0;

        count_local_average(b, GK, s, 2, i, j);

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
                if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_vert)) {
                      if (GK*s_hor < s_maindig* M1 && GK*s_hor < s_subdig* M1) Bvert[i*w+j]+=1;
                }
            } else if(M*s_hor > s_vert && M*GK*s_hor > s_maindig && M*GK*s_hor > s_subdig){
                       s_max = s_hor;
                       if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_hor)){
                            if (GK*s_vert < s_maindig* M1 && GK*s_vert < s_subdig* M1) Bhor[i*w+j]+=1;
                       }
            } else if( M*s_maindig > GK*s_hor && M*s_maindig > GK* s_vert && M*s_maindig > s_subdig){
                       s_max = s_maindig;
                       if ((s_max > AV*s*3) && (s_max > G_AV * av_diff_maindig)) {
                               if(GK*s_vert* M1 > s_subdig && GK*s_hor* M1 > s_subdig) Bmain[i*w+j]+=1;
                           }
            } else if( M*s_subdig > GK*s_hor && M*s_subdig > GK* s_vert && M*s_subdig > s_maindig) {
                s_max = s_subdig;
                if ((s_max > AV*s*3)  && (s_max > G_AV * av_diff_subdig) ) {
                        if(GK*s_vert* M1 > s_maindig && GK*s_hor* M1 > s_maindig) Bsub[i*w+j]+=1;
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
        } else {
            Map [ i*w + j] = 0;
        }
        }
    }

    delete Rhor;        delete Rvert;
    delete Rmain;       delete Rsub;
    delete Ghor;        delete Gvert;
    delete Gmain;       delete Gsub;
    delete Bhor;        delete Bvert;
    delete Bmain;       delete Bsub;

    delete Shor;        delete Svert;
    delete Smain;       delete Ssub;

return;
}
