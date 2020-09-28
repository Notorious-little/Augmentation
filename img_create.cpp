#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>       // Базовый набор функций cv
#include <opencv2/core/mat.hpp>        // Попробуем добавить это чтобы подключить Mat
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/highgui/highgui.hpp> // Взаимодействие с графическим интерфейсом
#include <cstring>
#include <cstdio>

using namespace cv;
using namespace std;

Mat autoContrast(const Mat img, const int quantil){
    
    assert(0<=quantil && quantil <=100);

    Mat img_autocontr = img;

    int gistR[256];
    int gistG[256];
    int gistB[256];

    int h = img_autocontr.rows;
    int w = img_autocontr.cols;
    
    for (int i=0; i<256; ++i){
        gistR[i] = 0;
        gistG[i] = 0;
        gistB[i] = 0;
    }

    for(int i=0; i<h; ++i){
        for(int j=0; j<w; ++j){
            ++gistR[ img_autocontr.at<Vec3b>(i, j)[0] ];
            ++gistG[ img_autocontr.at<Vec3b>(i, j)[1] ];
            ++gistB[ img_autocontr.at<Vec3b>(i, j)[2] ];
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
        qGmax(Gmax), qBmin(Bmin), qBmax(Bmax);     // quantil-Квантили распр-я для настройки новой интенсивности

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

    for(int i=0; i<h; ++i){
        for (int j=0; j<w; ++j){
            if (img_autocontr.at<Vec3b>(i, j)[0] > qRmin && img_autocontr.at<Vec3b>(i, j)[0] < qRmax )
                img_autocontr.at<Vec3b>(i, j)[0] = (int)( 255*(img_autocontr.at<Vec3b>(i, j)[0] - qRmin)/(qRmax-qRmin) ) % 256 ;

            if (img_autocontr.at<Vec3b>(i, j)[1] > qGmin && img_autocontr.at<Vec3b>(i, j)[1] < qGmax )
                img_autocontr.at<Vec3b>(i, j)[1] = (int)( 255*(img_autocontr.at<Vec3b>(i, j)[1] - qGmin)/(qGmax-qGmin) ) % 256 ;

            if (img_autocontr.at<Vec3b>(i, j)[2] > qBmin && img_autocontr.at<Vec3b>(i, j)[2] < qBmax )
                img_autocontr.at<Vec3b>(i, j)[2] = (int)( 255*(img_autocontr.at<Vec3b>(i, j)[2] - qBmin)/(qBmax-qBmin) ) % 256 ;
        }
    }   

return img_autocontr;
}


int main(){
    
    Mat img = imread ( "./mrb.bmp" , IMREAD_COLOR) ; 
    
    imshow( "Original", img);                     // Оконный интерфейс
    
    Mat autoc = autoContrast(img, 5);

    imshow( "Contrasted", autoc);                 // Оконный интерфейс

    waitKey(0);
    
    imwrite("./mrbinout.bmp", autoc);              // Вывод в файл (Ввод -imread)
    
return 0;
}

void labelText(Mat img){

    int h = img.rows;
    int w = img.cols;

    string text = "AUTOCONTRASTED";
    Point textOrg(0, h-7);                       //Местоположение  
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;  // Фонт (шрифт)
    double fontScale = 0.5;                      // Размер текста
    Scalar color(200, 100, 50);                  // Цвет

    putText(img, text, textOrg, fontFace, fontScale, color);
}

