#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <ctime>
#include <random>
#include "methods.h"
#include "OO_methods.h"
#include "gradient_map.h"
#include "map_filtrations.h"
#include "find_circle_around_point.h"
#include "v2_gradient_map.h"
#include "support.h"




struct TrackBarPar{

    const int alpha_slider_max_blur = 100;
    int alpha_slider_blur = 0;
    double alpha_blur = 3;
    double beta_blur = 10;

    const int alpha_slider_max_autoc = 20;
    int alpha_slider_autoc = 0;
    double alpha_autoc = 1;
    double beta_autoc = 1;

    double alpha = 1;
    double beta = 1;

    cv::Mat src1 =  cv::imread( "./barcode.bmp" , cv::IMREAD_COLOR) ;
    cv::Mat orig = src1.clone() ;
    cv::Mat src2 = src1.clone();
    cv::Mat dst = src1.clone();
    cv::Mat dst_autoc = src1.clone();

    int h = orig.rows;
    int w = orig.cols;

    int Map_size = (h+1)*(w+1);
    double *Map = new double[Map_size];

    int AV_slider = 0;
    int M1_slider = 0;
    int detnum_slider = 2;
    const int slider_max_AV = 40;
};


TrackBarPar Blur_Par;
TrackBarPar Autoc;
TrackBarPar Noize;
TrackBarPar GMC;




void drawTarget(IplImage* img, int x, int y, int radius)
{
    cv::Mat input_img = cv::cvarrToMat(img);

    std::default_random_engine rand_gen;
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    rand_gen.seed(seed);
    std::uniform_int_distribution<int> distribution(50, 70 );

    int Radius = distribution(rand_gen);

    cv::Point centerCircle(x,y);
    cv::Scalar colorCircle(Radius-30+20,Radius*2+20,Radius*3 );
    cv::circle(input_img, centerCircle, Radius, colorCircle, CV_FILLED);
    cvCircle(img, cvPoint(x, y), Radius, CV_RGB(250,0,0),1,8);

    cvCircle(img, cvPoint(x, y), radius, CV_RGB(250,0,0),1,8);
    cvLine(img, cvPoint(x-radius/2, y-radius/2), cvPoint(x+radius/2, y+radius/2),CV_RGB(250,0,0),1,8);
    cvLine(img, cvPoint(x-radius/2, y+radius/2), cvPoint(x+radius/2, y-radius/2),CV_RGB(250,0,0),1,8);
}


void drawTarget_2(IplImage* img, int x, int y, int radius){

    cv::Mat input_img = cv::cvarrToMat(img);

    int Radius = 20;

    cv::Point centerCircle(x,y);
    cv::Scalar colorCircle(Radius-30+20,Radius*2+20,Radius*3 );
    cvCircle(img, cvPoint(x, y), Radius, CV_RGB(250,0,0),1,8);

    cvCircle(img, cvPoint(x, y), radius, CV_RGB(250,0,0),1,8);
    cvLine(img, cvPoint(x-radius/2, y-radius/2), cvPoint(x+radius/2, y+radius/2),CV_RGB(250,0,0),1,8);
    cvLine(img, cvPoint(x-radius/2, y+radius/2), cvPoint(x+radius/2, y-radius/2),CV_RGB(250,0,0),1,8);
}


void myMouseCallback_IMAGE( int event, int x, int y, int flags, void* param )
{

    IplImage* img = (IplImage*) param;
    cv::Mat input_img = cv::cvarrToMat(img);

    switch( event ){

        case CV_EVENT_LBUTTONDOWN:
        {

            drawTarget(img, x, y, 10);
            cv::imshow("IMAGE", input_img);
            cv::waitKey(0);

            break;
        }

        case CV_EVENT_RBUTTONDOWN:
        {
            int N = 2;
            int Radius = find_circles(input_img, 2);

            if (Radius == 0) {
                break;
            } else {
                std::cout<<"HoughCircles returned error"<<
                std::endl<<std::flush ;
                break;
            }
        }

    }

return;
}


void show_BinaryContouredImage(const int GlobalTreshold){

    cv::Mat img = GMC.orig.clone();

    cv::Mat contoured = localContours(img, 6, 1.2, 2, 0, img.cols, 0, img.rows);

    cv::imshow( "Monochromed and contoured", contoured);

    cv::waitKey(0);

return;
}




static void on_GMC_trackbar( int, void* )
{
    GMC.src2 = GMC.orig.clone();

    if ((GMC.AV_slider > 0) && (GMC.alpha_slider_autoc > 0)
            && (GMC.M1_slider > 0)){

        localGradientMap(GMC.src2, GMC.Map,
                         GMC.alpha_slider_autoc,  0.3 + ((double)GMC.AV_slider/20),
                         0.3 + ((double)GMC.M1_slider/40), GMC.detnum_slider);

        GMC.src1 = draw_GM_contoured_img(GMC.src1, GMC.Map, GMC.Map_size);

        GMC.src1 = medianFilter_8UC1(GMC.src1, 4, GMC.orig.cols-4 , 4, GMC.orig.rows-4 );

        cv::imshow( "RGB-GRADIENT CONTOUR", GMC.src1 );
    }
}


void show_LocalContoured(){

    cv::Mat img = GMC.src1.clone();
    cv::Mat blured_img = blurByGaussMatrix(img, 1,  4, img.cols-4 , 4, img.rows-4 );
    cv::Mat AC_img = autoContrast(img, 1, 0, img.cols , 0, img.rows);

    GMC.src1 = AC_img.clone();
    GMC.orig = AC_img.clone();
    GMC.src2 = AC_img.clone();

    cv::namedWindow("RGB-GRADIENT CONTOUR", cv::WINDOW_AUTOSIZE) ;
    cv::imshow ("RGB-GRADIENT CONTOUR", AC_img );

    char TrackbarName[50] {"Global Treshold"};
    char Trackbar_2_Name[50] {"Local Average"};
    char Trackbar_3_Name[50] {"Parameter 'M' "};
    char Trackbar_4_Name[50] {" Channels "};


    cv::createTrackbar( TrackbarName, "RGB-GRADIENT CONTOUR", &GMC.alpha_slider_autoc,
                        GMC.alpha_slider_max_autoc, on_GMC_trackbar );

    cv::createTrackbar( Trackbar_2_Name, "RGB-GRADIENT CONTOUR", &GMC.AV_slider,
                        GMC.slider_max_AV, on_GMC_trackbar );

    cv::createTrackbar( Trackbar_3_Name, "RGB-GRADIENT CONTOUR", &GMC.M1_slider,
                        GMC.alpha_slider_max_blur, on_GMC_trackbar );

    cv::createTrackbar( Trackbar_4_Name, "RGB-GRADIENT CONTOUR", &GMC.detnum_slider,
                        3, on_GMC_trackbar );

    cv::waitKey(0);

    delete GMC.Map;

return;
}




void show_CannyContoured(){

    cv::Mat img = GMC.orig.clone();

    cv::Mat dst = img.clone();
    cvtColor( img , dst , cv::COLOR_BGR2GRAY );
    cv::Canny(img, dst, 10, 200, 3);
    cv::imshow( "Canny Contoured", dst);

    cv::waitKey(0);

return;
}




static void on_noized_trackbar( int, void* )
{
    Noize.src2 = Noize.orig.clone();
    Noize.src1 = gaussNoize(Noize.src2, (double)Noize.alpha_slider_autoc  );
    cv::imshow( "NOIZED", Noize.src1 );
}


void show_NoizedImages(){

    cv::Mat img = Noize.src1.clone();

    Noize.src1 = img.clone();
    Noize.orig = img.clone();
    Noize.src2 = gaussNoize(Noize.src1, 1);

    cv::namedWindow("NOIZED", cv::WINDOW_AUTOSIZE) ;
    cv::imshow ("NOIZED", Noize.src1 );

    char TrackbarName[50] {"Range"};

    cv::createTrackbar( TrackbarName, "NOIZED", &Noize.alpha_slider_autoc,
                        Noize.alpha_slider_max_autoc, on_noized_trackbar );

    cv::Mat SaltPaperNoized_image = salt_paperNoize( img, 8083647);
    cv::imshow( "Salt-Papper-Noized", SaltPaperNoized_image);

    cv::waitKey(0);

return;
}




static void on_autoc_trackbar( int, void* )
{
    Autoc.beta_autoc = 1;
    Autoc.src2 = Autoc.orig.clone();
    Autoc.src1 = autoContrast(Autoc.src2, ((int)Autoc.alpha_slider_autoc ),
                              0, Autoc.src2.cols , 0, Autoc.src2.rows);
    cv::imshow( "AUTOCONTRASTED", Autoc.src1 );
}


void show_AutocontrImage(){

    cv::Mat img = Autoc.src1.clone();

    Autoc.src1 = img.clone();
    Autoc.orig = img.clone();
    Autoc.src2 = autoContrast(Autoc.src1, 5,
                               0, Autoc.src1.cols , 0, Autoc.src1.rows);

    cv::namedWindow("AUTOCONTRASTED", cv::WINDOW_AUTOSIZE) ;
    cv::imshow ("AUTOCONTRASTED", Autoc.src1 );

    char TrackbarName[50] {"Quantil"};

    cv::createTrackbar( TrackbarName, "AUTOCONTRASTED", &Autoc.alpha_slider_autoc,
                        Autoc.alpha_slider_max_autoc, on_autoc_trackbar );

    cv::waitKey(0);

return;
}




static void on_blur_trackbar( int, void* )
{
    Blur_Par.alpha_blur = (double) Blur_Par.alpha_slider_blur
                                  /Blur_Par.alpha_slider_max_blur ;
    Blur_Par.beta_blur = ( 1.0 - Blur_Par.alpha_blur );

    cv::addWeighted( Blur_Par.src1, Blur_Par.beta_blur,
                     Blur_Par.src2, Blur_Par.alpha_blur, 0.0, Blur_Par.dst);
    cv::imshow( "BLURED", Blur_Par.dst );
}


void show_BluredImage(){

    Blur_Par.src2 = blurByGaussMatrix(Blur_Par.src1, 15, 0, Blur_Par.src1.cols-4 , 0, Blur_Par.src1.rows-4);

    cv::namedWindow("BLURED", cv::WINDOW_AUTOSIZE) ;
    cv::imshow ("BLURED", Blur_Par.src1 );

    char TrackbarName[50] {"Blur power"};

    cv::createTrackbar( TrackbarName, "BLURED", &Blur_Par.alpha_slider_blur,
                        Blur_Par.alpha_slider_max_blur, on_blur_trackbar );

    cv::waitKey(0);

return;
}




void show_WideWindowContoured(){

    cv::Mat img = GMC.orig.clone();

    int Map_size = (img.cols)*(img.rows);
    double *Map = new double[Map_size] {0};

    cv::Mat blured_img = blurByGaussMatrix(img, 4, 4, img.cols-4 , 4, img.rows-4);
    cv::Mat AC_img = autoContrast(img, 5,  0, img.cols , 0, img.rows);
    wide_Window_map(AC_img, Map, 2);
    cv::Mat map_img = draw_GM_contoured_img(img, Map, Map_size);
    cv::Mat medianed = medianFilter_8UC1(map_img, 4, img.cols-4 , 4, img.rows-4);
    cv::imshow("Wide Window Gradient-Contour", medianed);

    cv::waitKey(0);

    delete Map;

return;
}




void show_OriginalImage(cv::Mat &img){

    IplImage* image = cvCreateImage(cvSize(img.cols, img.rows), 8, 3);
    IplImage ipltemp = img;
    cvCopy(&ipltemp, image);

    cv::imshow("IMAGE", img);
    cvSetMouseCallback( "IMAGE", myMouseCallback_IMAGE, (void*) image);

    cv::waitKey(0);

return;
}
