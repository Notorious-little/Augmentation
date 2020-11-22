#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "barcode.h"


cv::Mat autoContrast(const cv::Mat &img, const int quantil, const int L, const int R, const int U, const int D){

    assert(0 <= quantil && quantil <= 100);

    if (quantil == 0){
        return img;
    }

    cv::Mat img_autocontr = img.clone();

    int gistR[256] = {0};
    int gistG[256] = {0};
    int gistB[256] = {0};

    int h = D-U;
    int w = R-L;

    for(int i = U; i < D; ++i){
        for(int j = L; j < R; ++j){
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


    for(int i = U; i < D; ++i){
        for (int j = L; j < R; ++j){
            if (img_autocontr.at<cv::Vec3b>(i, j)[0] > qRmin && img_autocontr.at<cv::Vec3b>(i, j)[0] < qRmax )
                img_autocontr.at<cv::Vec3b>(i, j)[0] =
                        (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[0] - qRmin)/(qRmax-qRmin) ) % 255 ;

            if (img_autocontr.at<cv::Vec3b>(i, j)[1] > qGmin && img_autocontr.at<cv::Vec3b>(i, j)[1] < qGmax )
                img_autocontr.at<cv::Vec3b>(i, j)[1] =
                        (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[1] - qGmin)/(qGmax-qGmin) ) % 255 ;

            if (img_autocontr.at<cv::Vec3b>(i, j)[2] > qBmin && img_autocontr.at<cv::Vec3b>(i, j)[2] < qBmax )
                img_autocontr.at<cv::Vec3b>(i, j)[2] =
                        (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[2] - qBmin)/(qBmax-qBmin) ) % 255 ;
        }
    }

return img_autocontr;
}




void labelText(cv::Mat img, const int Amount){

    int h = img.rows;
    int w = img.cols;

    std::string text = "AMOUNT: ";
    text += std::to_string(Amount);
    cv::Point textOrg(0, 50);                       // Местоположение
    int fontFace = cv::FONT_HERSHEY_TRIPLEX ;        // Фонт (шрифт)
    double fontScale = 1.5;                          // Размер текста
    cv::Scalar color(0, 0, 255);                     // Цвет

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





cv::Mat blurByGaussMatrix(const cv::Mat &input_img, const int blurpower,
                           const int L, const int R, const int U, const int D){

    cv::Mat bluring_img = input_img.clone();

    const int n = 2*blurpower+1;
    const int h = input_img.rows;
    const int w = input_img.cols;

    double* Matrix = new double [n*n];

    slidingNormalMatrixCompute( Matrix , blurpower );

    for (int i = blurpower + U; i < D - blurpower; ++i){
        for (int j = blurpower+L; j < R - blurpower; ++j){
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

    delete Matrix;

return bluring_img;
}




cv::Mat blur_monochrome_ByGauss(const cv::Mat &input_mono_img, const int blurpower,
                           const int L, const int R, const int U, const int D){

    cv::Mat bluring_mono_img = input_mono_img.clone();

    const int n = 2*blurpower+1;
    const int h = input_mono_img.rows;
    const int w = input_mono_img.cols;

    double* Matrix = new double [n*n];

    slidingNormalMatrixCompute( Matrix , blurpower );

    for (int i = blurpower + U; i < D - blurpower; ++i){
        for (int j = blurpower+L; j < R - blurpower; ++j){

            double intence = 0;

            for (int k = i - blurpower; k < i + blurpower + 1; ++k){
                for (int l = j - blurpower; l < j + blurpower +1; ++l){

                    intence += ( input_mono_img.at<uchar>(k,l) *
                            Matrix[(k - i + blurpower) * n + (l - j + blurpower)]);

                }
            }

        bluring_mono_img.at<uchar>(i,j) = (int)intence;

        }
    }

    delete Matrix;

return bluring_mono_img;
}




IplImage* Lapl_of_Gauss_colored(IplImage* frame, const double Param,       // Param in [0;1]
                           const int L, const int R, const int U, const int D){

    const cv::Mat input_mono_img = cv::cvarrToMat(frame);

    cv::Mat LoG_mono_img = input_mono_img.clone();

    double a = Param;
    double* Matrix = new double [9] {-a, a-1, -a, a-1, a+5, a-1, -a, a-1, -a};

    for (int i = 1 + U; i < D - 1; ++i){
        for (int j = 1+L; j < R - 1; ++j){

            double intence_Red = 0;
            double intence_Green = 0;
            double intence_Blue = 0;

            for (int k = i - 1; k < i + 2; ++k){
                for (int l = j - 1; l < j + 2; ++l){

                    intence_Red += ( input_mono_img.at<cv::Vec3b>(k,l)[0] *
                            Matrix[(k - i + 1) * 3 + (l - j + 1)]);

                    intence_Green += ( input_mono_img.at<cv::Vec3b>(k,l)[1] *
                            Matrix[(k - i + 1) * 3 + (l - j + 1)]);

                    intence_Blue += ( input_mono_img.at<cv::Vec3b>(k,l)[2] *
                            Matrix[(k - i + 1) * 3 + (l - j + 1)]);

                }
            }

        LoG_mono_img.at<cv::Vec3b>(i,j)[0] = (int)intence_Red;
        LoG_mono_img.at<cv::Vec3b>(i,j)[1] = (int)intence_Green;
        LoG_mono_img.at<cv::Vec3b>(i,j)[2] = (int)intence_Blue;

        }
    }
cv::imshow("TEST", LoG_mono_img);
cv::waitKey(0);
    delete Matrix;

    IplImage* image_return;
    image_return = cvCreateImage(cvSize(LoG_mono_img.cols,LoG_mono_img.rows), 8, 3);  // (..., 8 , 1) ??
    IplImage ipltemp=LoG_mono_img;
    cvCopy(&ipltemp,image_return);

return image_return;
}


/* #define RAND_MAX 2147483647*/
double exp_rand(void){

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




cv::Mat gaussNoize( const cv::Mat &input_img, const double noize_range){

    if ( abs(noize_range) < 1e-9){
        return input_img;
    }

    cv::Mat noizing_img = input_img.clone();

    int w = noizing_img.cols;
    int h = noizing_img.rows;
        
    for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){

                double e = exp_rand();
                e /= 100;
                double multiplier = (double)(noize_range)/10;

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




cv::Mat salt_paperNoize( const cv::Mat &input_img, const int noize_range){

        cv::Mat noizing_img = input_img.clone();

        int w = noizing_img.cols;
        int h = noizing_img.rows;

        for (int i = 0; i < h; ++i){
                for (int j = 0; j < w; ++j){

                    int a = rand();

                    if( a < noize_range ){
                        noizing_img.at<cv::Vec3b>(i,j)[0] = 0;
                        noizing_img.at<cv::Vec3b>(i,j)[1] = 0;
                        noizing_img.at<cv::Vec3b>(i,j)[2] = 0;
                    }

                    if( a > (RAND_MAX - noize_range) ){
                        noizing_img.at<cv::Vec3b>(i,j)[0] = 255;
                        noizing_img.at<cv::Vec3b>(i,j)[1] = 255;
                        noizing_img.at<cv::Vec3b>(i,j)[2] = 255;
                    }

                }
        }

return noizing_img;
}



cv::Mat localContours(const cv::Mat &input_img,
                      const double G_AV, const double AV, const double M,
                      const int L, const int R, const int U, const int D){

    cv::Mat output_img( input_img.rows , input_img.cols , CV_8UC1 );
    cv::Mat b1 = input_img.clone();
    cvtColor( input_img , b1 , cv::COLOR_BGR2GRAY );

    cv::Mat b;
    b1.convertTo(b, CV_8UC1);

    const double GK = sqrt(2);

    const int h = input_img.rows;
    const int w = input_img.cols;

    long double av_diff_vert(0), av_diff_hor(0), av_diff_maindig(0), av_diff_subdig(0);

    for (int i = U; i < D; ++i ){

    long double tmp_diff_vert(0), tmp_diff_hor(0),
                tmp_diff_maindig(0), tmp_diff_subdig(0);

        for (int j = L; j < R; ++j ){

            tmp_diff_vert += abs(b.at<uchar>(i-1,j) - b.at<uchar>(i+1,j));

            tmp_diff_hor += abs(b.at<uchar>(i,j-1) - b.at<uchar>(i,j+1));

            tmp_diff_maindig += (abs(b.at<uchar>(i-1,-1) - b.at<uchar>(i+1,j+1))) / sqrt(2);

            tmp_diff_subdig += (abs(b.at<uchar>(i-1,j+1) - b.at<uchar>(i+1,j-1))) / sqrt(2);
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


    for (int i = U; i < D; ++i ){
        for (int j = L; j < R; ++j ){

        output_img.at<uchar>(i,j) = 0;

        double s = 0;

            for(int k = j - 1; k <= j+1; ++k){
                s += abs (b.at<uchar>(i,k) - b.at<uchar>(i-2,k))*GK;
                s += abs (b.at<uchar>(i+1,k) - b.at<uchar>(i-1,k))*GK;
                s += abs (b.at<uchar>(i+2,k) - b.at<uchar>(i,k))*GK;
            }

            for(int k = i - 1; k <= i+1; ++k){
                s += abs (b.at<uchar>(k,j) - b.at<uchar>(k,j-2))*GK;
                s += abs (b.at<uchar>(k,j+1) - b.at<uchar>(k,j-1))*GK;
                s += abs (b.at<uchar>(k,j+2) - b.at<uchar>(k,j))*GK;
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
                s_vert += abs(b.at<uchar>(i-1,j+k) - b.at<uchar>(i+1,j+k))*GK;
            }

            for(int k = -1; k <= 1; ++k){
                s_hor += abs(b.at<uchar>(i+k,j-1) - b.at<uchar>(i+k,j+1))*GK;
            }

            for(int k = 0; k <= 2; ++k){
                s_maindig += abs(b.at<uchar>(i-2+k,j+k) - b.at<uchar>(i+k,j-2+1));
            }

            for(int k = 0; k <= 2; ++k){
                s_subdig += abs(b.at<uchar>(i-2+k,j-k) - b.at<uchar>(i+k,j+2-k));
            }


            if ( s_vert > s_hor && s_vert > s_maindig && s_vert > s_subdig){
                s_max = s_vert;
                if ((s_max > s*3*AV) && (s_max > G_AV * av_diff_vert)) {
                    if (s_hor < s_maindig && s_hor < s_subdig)
                        for(int c = -1 ; c < 2; ++c){
                            output_img.at<uchar>(i,j+c) = 255;
                        }
                }
            } else if(s_hor >= s_vert && s_hor >= s_maindig && s_hor >= s_subdig){
                       s_max = s_hor;
                       if ((s_max > s*3*AV) && (s_max > G_AV * av_diff_hor)) {
                           if (s_vert < s_maindig && s_vert < s_subdig)
                               for(int c = -1 ; c < 2; ++c){
                                   output_img.at<uchar>(i+c,j) = 255;
                               }
                       }
            } else if(s_maindig > s_hor && s_maindig > s_vert &&
                      s_maindig > s_subdig && ((s_max > s*3*AV) && (s_max > G_AV * av_diff_maindig))){

                           s_max = s_maindig;

                           if(s_vert > s_subdig && s_hor > s_subdig)
                               for(int c = -1 ; c < 2; ++c){
                                   output_img.at<uchar>(i+c,j+c) = 255;
                               }
            } else {
                s_max = s_subdig;
                if ((s_max > s*3*AV) && (s_max > G_AV * av_diff_subdig)) {
                    if(s_vert > s_maindig && s_hor > s_maindig)
                        for(int c = -1 ; c < 2; ++c){
                            output_img.at<uchar>(i+c,j-c) = 255;
                        }
                }
              }
        }
    }

return output_img;
}




cv::Mat medianFilter_8UC1 (const cv::Mat &input_img, const int &L,
                           const int &R, const int &U, const int &D){         // Против изолированых белых точек в контуре

    cv::Mat output_img = input_img.clone();

    output_img.convertTo(output_img, CV_8UC1);

    int max = 50;

    int h = output_img.rows;
    int w = output_img.cols;

    for (int i = U; i < D; ++i){
        for (int j = L; j < R; ++j){

            if (input_img.at<uchar>(i,j) > max){

              if(( ( (input_img.at<uchar>(i,j-1)  <= max) &&
                      (input_img.at<uchar>(i,j+1)  <= max) ) &&
                    ( (input_img.at<uchar>(i-1,j)   <= max) &&
                      (input_img.at<uchar>(i+1,j)   <= max) ) &&
                    ( (input_img.at<uchar>(i-1,j-1) <= max) &&
                      (input_img.at<uchar>(i+1,j+1) <= max) ) &&
                    ( (input_img.at<uchar>(i+1,j-1) <= max) &&
                      (input_img.at<uchar>(i-1,j+1) <= max)) )) {

                    output_img.at<uchar>(i,j) = 0;

              } else{
                  output_img.at<uchar>(i,j) = 255;
              }
            } else {
                output_img.at<uchar>(i,j) = 0;
            }
        }
    }

return output_img;
}




cv::Mat binarization(const cv::Mat &input_mono_img,              // Monochrome-CV_8UC1-image inputed!
                     const int L, const int R, const int U, const int D){
    cv::Mat output_img( input_mono_img.rows , input_mono_img.cols , CV_8UC1 );


    for (int i = U+1; i < D; i += 1 ){
        for (int j = L+1; j < R; j += 1 ){
                 output_img.at<uchar>(i,j) = 255;
        }
    }

    int AV = count_AT( input_mono_img, L+10, U+10);
    int G_AV =  count_Global_AT( input_mono_img, (U+D)/2, (L+R)/2) ;

    int *averages_map = new int[ 2*(R-L)*(D-U) ]{0};

    int vert_cuts = 24, hor_cuts = 32;

    for (int v = 0; v < vert_cuts; ++v){
        for (int h = 0; h < hor_cuts; ++h){

            int av_v_h = count_AT( input_mono_img,

                                          (int)(U + (0.5+h)*(D-U)/hor_cuts) , (int)(L + (0.5+v)*(R-L)/vert_cuts));

            for (int i = U + (int)(v*(D-U)/vert_cuts); i < U + (int)((1+v)*(D-U)/vert_cuts); ++i){
                for (int j = L + (int)(h*(R-L)/hor_cuts); j < L + (int)((h+1)*(R-L)/hor_cuts); ++j){

                    if ((v<2) || (v>21) || (h<3) || (h>28))
                        av_v_h -= 15;

                    averages_map[i*(R-L) + j] = av_v_h;

                }
            }
        }
    }

    for (int i = U; i < D; i += 1 ){
        for (int j = L; j < R; j += 2 ){

            if ((input_mono_img.at<uchar>(i,j) < averages_map[i*(R-L) + j]) &&
                (input_mono_img.at<uchar>(i,j) < G_AV ) ){

                output_img.at<uchar>(i,j) = 0;

            } else {
                output_img.at<uchar>(i,j) = 255;
            }

        }
    }

    delete averages_map;

return output_img;
}




cv::Mat horizontal_dilatate (const cv::Mat &input_img, const int L, const int R,
                          const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U; i < D; ++i ){
        for (int j = L; j < R; ++j ) {

            if( (input_img.at<uchar>(i,j-1) == 0 ) || (input_img.at<uchar>(i,j+1) == 0 ) ||
                (input_img.at<uchar>(i-1,j) == 0 ) || (input_img.at<uchar>(i+1,j) == 0 ) ||
                (input_img.at<uchar>(i,j-2) == 0 ) || (input_img.at<uchar>(i,j+2) == 0 ) ){

                    output_img.at<uchar>(i,j) = 0;

            } else {

                output_img.at<uchar>(i,j) = 255;
            }

        }
    }

return output_img;
}




cv::Mat horizontal_dilatate_leftside (const cv::Mat &input_img, const int L, const int R,
                          const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U; i < D; ++i ){
        for (int j = L; j < (L+R)/2 ; ++j ) {

            if( (input_img.at<uchar>(i+1,j) == 0 ) || (input_img.at<uchar>(i-1,j) == 0 ) ||
                (input_img.at<uchar>(i,j-2) == 0 ) || (input_img.at<uchar>(i,j-1) == 0 ) ||
                (input_img.at<uchar>(i,j) == 0 ) )  {

                    output_img.at<uchar>(i,j) = 0;

            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}




cv::Mat horizontal_dilatate_rightside (const cv::Mat &input_img, const int L, const int R,
                              const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U; i < D; ++i ){
        for (int j = (L+R)/2; j < R; ++j ) {

            if( (input_img.at<uchar>(i+1,j) == 0 ) || (input_img.at<uchar>(i-1,j) == 0 ) ||
                (input_img.at<uchar>(i,j+2) == 0 ) || (input_img.at<uchar>(i,j+1) == 0 ) ||
                (input_img.at<uchar>(i,j) == 0 ) )  {

                    output_img.at<uchar>(i,j) = 0;

            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}



cv::Mat rightside_dilatate (const cv::Mat &input_img, const int L, const int R,
                              const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U; i < D; ++i ){
        for (int j = (L+R)/2; j < R; ++j ) {

            if( (input_img.at<uchar>(i,j+1) == 0 ) ||
                (input_img.at<uchar>(i,j) == 0 ) )  {

                    output_img.at<uchar>(i,j) = 0;

            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}




cv::Mat leftside_dilatate (const cv::Mat &input_img, const int L, const int R,
                           const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U; i < D; ++i ){
        for (int j = L; j < (L+R)/2 ; ++j ) {

            if( (input_img.at<uchar>(i,j-1) == 0 ) ||
                (input_img.at<uchar>(i,j) == 0 ) )  {

                    output_img.at<uchar>(i,j) = 0;

            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}



cv::Mat classic_dilatate(const cv::Mat &input_img, const int L, const int R,
                         const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U+1; i < D-1; ++i ){
        for (int j = L+1; j < R-1; ++j ){

            if((input_img.at<uchar>(i-1,j-1) == 0 ) || (input_img.at<uchar>(i-1,j) == 0 ) ||
               (input_img.at<uchar>(i-1,j+1) == 0 ) || (input_img.at<uchar>(i,j-1) == 0 ) ||
               (input_img.at<uchar>(i,j+1) == 0 ) || (input_img.at<uchar>(i+1,j-1) == 0 ) ||
               (input_img.at<uchar>(i+1,j) == 0 ) || (input_img.at<uchar>(i+1,j+1) ==0 )) {

                    output_img.at<uchar>(i,j) = 0;
            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}




cv::Mat classic_erosion(const cv::Mat &input_img, const int L, const int R,
                        const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U+1; i < D-1; ++i ){
        for (int j = L+1; j < R-1; ++j ){

            if((input_img.at<uchar>(i-1,j-1) == 0 ) && (input_img.at<uchar>(i-1,j) == 0 ) &&
               (input_img.at<uchar>(i-1,j+1) == 0 ) && (input_img.at<uchar>(i,j-1) == 0 ) &&
               (input_img.at<uchar>(i,j+1) == 0 ) && (input_img.at<uchar>(i+1,j-1) == 0 ) &&
               (input_img.at<uchar>(i+1,j) == 0 ) && (input_img.at<uchar>(i+1,j+1) ==0 )) {

                    output_img.at<uchar>(i,j) = 0;
            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}



cv::Mat vertical_erosion(const cv::Mat &input_img, const int L, const int R,
                  const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U+1; i < D-1; ++i ){
        for (int j = L+1; j < R-1; ++j ){

            if((input_img.at<uchar>(i-2,j) == 0 ) && (input_img.at<uchar>(i+2,j) == 0 ) &&
               (input_img.at<uchar>(i-1,j) == 0 ) && (input_img.at<uchar>(i+1,j) == 0 ) &&
               ((input_img.at<uchar>(i,j+1) == 0 ) || (input_img.at<uchar>(i,j-1) == 0 ))) {

                    output_img.at<uchar>(i,j) = 0;
            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}




cv::Mat horizontal_erosion(const cv::Mat &input_img, const int L, const int R,
                  const int U, const int D){

    cv::Mat output_img = input_img.clone();

    for (int i = U+1; i < D-1; ++i ){
        for (int j = L+1; j < R-1; ++j ){

            if((input_img.at<uchar>(i,j-2) == 0 ) && (input_img.at<uchar>(i,j+2) == 0 ) &&
               (input_img.at<uchar>(i,j-1) == 0 ) && (input_img.at<uchar>(i,j+1) == 0 )) {

                    output_img.at<uchar>(i,j) = 0;
            } else {

                output_img.at<uchar>(i,j) = 255;
            }
        }
    }

return output_img;
}



cv::Mat draw_barcode(const cv::Mat &input_img, const int L, const int R,
                     const int U, const int D ){

    cv::Mat output_img = input_img.clone();

    int M = (U+D)/2;
    int mid = 0;

    for (int i = L; i < R; ++i){
        mid += input_img.at<cv::Vec3b>( M, i)[0];
        mid += input_img.at<cv::Vec3b>( M, i)[1];
        mid += input_img.at<cv::Vec3b>( M, i)[2];

    }

    mid = (int)( 1.0 * (double)mid / (double)(3*R-3*L));

    for (int i = L; i < R; ++i){

        if (    ((input_img.at<cv::Vec3b>( M, i)[0] > mid ) &&
                        (input_img.at<cv::Vec3b>( M, i)[1] > mid )) ||
                ((input_img.at<cv::Vec3b>( M, i)[1] > mid ) &&
                            (input_img.at<cv::Vec3b>( M, i)[2] > mid )) ||
                ((input_img.at<cv::Vec3b>( M, i)[0] > mid ) &&
                            (input_img.at<cv::Vec3b>( M, i)[2] > mid )) ) {
            for (int w = -10; w <= 10; ++w){
                output_img.at<cv::Vec3b>( M+w, i)[0] = 255;
                output_img.at<cv::Vec3b>( M+w, i)[1] = 255;
                output_img.at<cv::Vec3b>( M+w, i)[2] = 255;
            }

        } else {

            for (int w = -10; w <= 10; ++w){
                output_img.at<cv::Vec3b>( M+w, i)[0] = 0;
                output_img.at<cv::Vec3b>( M+w, i)[1] = 0;
                output_img.at<cv::Vec3b>( M+w, i)[2] = 0;
            }
        }
    }

return output_img;
}
