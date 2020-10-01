#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>       // Базовый набор функций cv
#include <opencv2/core/mat.hpp>        // Попробуем добавить это чтобы подключить Mat
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/highgui/highgui.hpp> // Взаимодействие с графическим интерфейсом
#include <cstring>
#include <cstdio>
#include <cmath>

# define M_PI           3.14159265358979323846

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
                img_autocontr.at<cv::Vec3b>(i, j)[0] = (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[0] - qRmin)/(qRmax-qRmin) ) % 256 ;

            if (img_autocontr.at<cv::Vec3b>(i, j)[1] > qGmin && img_autocontr.at<cv::Vec3b>(i, j)[1] < qGmax )
                img_autocontr.at<cv::Vec3b>(i, j)[1] = (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[1] - qGmin)/(qGmax-qGmin) ) % 256 ;

            if (img_autocontr.at<cv::Vec3b>(i, j)[2] > qBmin && img_autocontr.at<cv::Vec3b>(i, j)[2] < qBmax )
                img_autocontr.at<cv::Vec3b>(i, j)[2] = (int)( 255*(img_autocontr.at<cv::Vec3b>(i, j)[2] - qBmin)/(qBmax-qBmin) ) % 256 ;
        }
    }   

return img_autocontr;
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






int main(){
    
    cv::Mat img = cv::imread ( "./mrb.bmp" , cv::IMREAD_COLOR) ;
    
    cv::imshow( "Original", img);                     
    
    cv::Mat autoc = autoContrast(img, 5);

    cv::imshow( "Contrasted", autoc);                 

    cv::Mat blured_img = blurByGaussMatrix(img, 5);

    cv::imshow( "Blured", blured_img);

    cv::waitKey(0);
    
    cv::imwrite("./mrbinout.bmp", autoc);             
    
return 0;
}
