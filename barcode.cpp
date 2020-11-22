#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "methods.h"
#include "map_filtrations.h"
#include "barcode_in_zone.h"
#include "support.h"




void draw_markup(IplImage* frame, int width, int height){

    int Radius = 20;
    int radius = 12;
    int x = (int) width/2;
    int y = (int) height/2;

    cv::Point center_point(x,y);
    cv::Scalar draw_color(0,0,255 );

    cvLine(frame, cvPoint(x - (int)(width/3), y - (int)(height/3.5)),
           cvPoint(x-(int)(width/3), y + (int)(height/3.5)), CV_RGB(250,0,0),3,8);

    cvLine(frame, cvPoint(x + (int)(width/3), y - (int)(height/3.5)),
           cvPoint(x + (int)(width/3), y + (int)(height/3.5)), CV_RGB(250,0,0),3,8);

    cvLine(frame, cvPoint(x + (int)(width/3), y - (int)(height/3.5)),
           cvPoint(x + (int)(width/3) - 40, y - (int)(height/3.5)),CV_RGB(250,0,0),3,8);

    cvLine(frame, cvPoint(x + (int)(width/3), y + (int)(height/3.5)),
           cvPoint(x + (int)(width/3) - 40, y + (int)(height/3.5)),CV_RGB(250,0,0),3,8);

    cvLine(frame, cvPoint(x - (int)(width/3), y - (int)(height/3.5)),
           cvPoint(x - (int)(width/3) + 40, y - (int)(height/3.5)),CV_RGB(250,0,0),3,8);

    cvLine(frame, cvPoint(x - (int)(width/3), y + (int)(height/3.5)),
           cvPoint(x - (int)(width/3) + 40, y + (int)(height/3.5)),CV_RGB(250,0,0),3,8);
}




void draw_circles(IplImage* frame, int x, int y){

    int Radius = 20;
    int radius = 12;

    cv::Point center_point(x,y);
    cv::Scalar draw_color(0,0,255 );
    cvCircle(frame, cvPoint(x, y), Radius, CV_RGB(250,0,0),3,8);
    cvCircle(frame, cvPoint(x, y), radius, CV_RGB(250,0,0),1,8);
}



void draw_corner_circles(IplImage* frame, int x, int y){

    int Radius = 8;
    int radius = 4;

    cv::Point center_point(x,y);

    cvCircle(frame, cvPoint(x, y), Radius, CV_RGB(0,250,0),3,8);
    cvCircle(frame, cvPoint(x, y), radius, CV_RGB(0,250,0),1,8);
}



void draw_corner2_circles(IplImage* frame, int x, int y){

    int Radius = 8;
    int radius = 4;

    cv::Point center_point(x,y);

    cvCircle(frame, cvPoint(x, y), Radius, CV_RGB(0,0,250),3,8);
    cvCircle(frame, cvPoint(x, y), radius, CV_RGB(0,0,250),1,8);
}



void show_persp_transformed(IplImage* in_img,
         const int L, const int R, const int U, const int D,
         int a1, int b1, int a2, int b2,
         int a3, int b3, int a4, int b4) {

    IplImage *frame = in_img, *dst=0;

    CvPoint2D32f srcQuad[4], dstQuad[4];  // (x,y) - points
    CvMat* warp_matrix = cvCreateMat(3, 3, CV_32FC1);

    dst = cvCloneImage(frame);

    //  frame = Lapl_of_Gauss_colored(frame, 0.5, L, R, U, D);

    srcQuad[0].x = a1;           //src Top left
    srcQuad[0].y = b1;
    srcQuad[1].x = a2;           //src Top right
    srcQuad[1].y = b2;
    srcQuad[2].x = a3;           //src Bottom left
    srcQuad[2].y = b3;
    srcQuad[3].x = a4;           //src Bot right
    srcQuad[3].y = b4;

    dstQuad[0].x = a1;           //dst Top left
    dstQuad[0].y = b1;
    dstQuad[1].x = a1;           //dst Top right
    dstQuad[1].y = b3;
    dstQuad[2].x = a3;           //dst Bottom left
    dstQuad[2].y = b3;
    dstQuad[3].x = a3;           //dst Bot right
    dstQuad[3].y = b1;


    #define  CV_WARP_INVERSE_MAP  16

    cvGetPerspectiveTransform(srcQuad, dstQuad, warp_matrix);

    cvWarpPerspective(frame, dst, warp_matrix);

    cv::Mat input_img = cv::cvarrToMat(dst);

    input_img = draw_barcode(input_img, a1 - 20, a3 + 20, b1, b3);

    cv::imshow("Perspective-Corrected", input_img);
    //cvShowImage( "Perspective-Corrected", dst );

    char c = cvWaitKey(0);
    if ( c == '27'){
        return;
    }

    cvReleaseMat(&warp_matrix);
    cvReleaseImage(&dst);

    cvDestroyAllWindows();
    return;
}




int count_AT ( cv::Mat img, int x, int y){

    cv::Mat b1 = img.clone();
    cv::Mat b;
    b1.convertTo(b, CV_8UC1);

    int mid = 0, max = 0, min = 255;

    for (int i = -5; i < 5; ++i){
        mid += b.at<uchar>(x+i, y);
    }

    for (int i = -10; i < 10; ++i){
        mid += b.at<uchar>(x, y+i);

        if ( b.at<uchar>(x, y+i) < min)
            min = b.at<uchar>(x, y+i);

        if ( b.at<uchar>(x, y+i) > max)
            max = b.at<uchar>(x, y+i);

    }

    for (int i = -5; i < 5; ++i){
        mid += b.at<uchar>(x+i, y+i);
    }

    for (int i = -5; i < 5; ++i){
        mid += b.at<uchar>(x+i, y-i);
    }

    mid = (int)(mid/50);

    for (int i = -7; i <= 7; ++i){

        if ( b.at<uchar>(x, y+i) < min)
            min = b.at<uchar>(x, y+i);

        if ( b.at<uchar>(x, y+i) > max)
            max = b.at<uchar>(x, y+i);

    }

    if (((mid-min) > 8) && ((max-mid) > 8) ) {
        return mid+15;
    } else {
        return mid-25;
    }
}


int count_Global_AT ( cv::Mat img, int x, int y){

    cv::Mat b1 = img.clone();
    cv::Mat b;
    b1.convertTo(b, CV_8UC1);

    int mid = 0;

    for (int i = -60; i < 60; ++i){
        mid += b.at<uchar>(x+i, y);
    }

    for (int i = -120; i < 120; ++i){
        mid += b.at<uchar>(x, y+i);
    }

    for (int i = -30; i < 30; ++i){
        mid += b.at<uchar>(x+i, y+i);
    }

    for (int i = -30; i < 30; ++i){
        mid += b.at<uchar>(x+i, y-i);
    }

    mid = (int)(mid/480);

return mid;
}


int count_local_AT ( cv::Mat b, int x, int y){


    int mid = 0;

    for (int i = -10; i < 10; ++i){
        for (int j = -10; j < 10; ++j){
            mid += b.at<uchar>(x+i, y+j);
        }
    }


return (mid/400);
}


int barcode_show_out(IplImage &frame, const int width, const int height){

    int x = (int) width/2;
    int y = (int) height/2;
    int L = x - (int)(width/3), R = x + (int)(width/3),
        U = y - (int)(height/3.5), D = y + (int)(height/3.5);

    cv::Mat img = cv::cvarrToMat(&frame);

    img = autoContrast(img, 5,  L, R, U, D);  // Should be not-linear autocontr.

    cv::Mat b1 = img.clone();
    cvtColor( img , b1 , cv::COLOR_BGR2GRAY );

    cv::Mat b ;
    b1.convertTo(img, CV_8UC1);

    img = blur_monochrome_ByGauss(img, 2 , L, R, U, D);

    cv::Mat bin_img = binarization(img, L, R, U, D );


    int Line_L{0}, Line_R{0}, Line_U{0}, Line_D{0};

    for( int i = 0; i < 3; ++i){
        bin_img = classic_dilatate(bin_img, L, R, U, D);
    }


    for( int i = 0; i < 5; ++i){
        bin_img = horizontal_dilatate_leftside(bin_img, L, R, U, D);
        bin_img = horizontal_dilatate_rightside(bin_img, L, R, U, D);
    }


    for( int i = 0; i < 12; ++i){
        bin_img = rightside_dilatate(bin_img, L, R, U, D);
        bin_img = leftside_dilatate(bin_img, L, R, U, D);
    }

    for( int i = 0; i < 2; ++i){
        bin_img = classic_erosion(bin_img, L, R, U, D);
        bin_img = vertical_erosion(bin_img, L, R, U, D);
        bin_img = horizontal_erosion(bin_img, L, R, U, D);
    }

    bin_img = vertical_erosion(bin_img, L, R, U, D);
    bin_img = vertical_erosion(bin_img, L, R, U, D);

    int Square = 0;

    bin_img = central_connected_component (bin_img, L, R, U, D, &Square);

    if ((Square < 80000) && ( Square > 25000)){

        int angles_ok = detect_min_rectangle(&frame, bin_img, Line_L, Line_R,
                             Line_U, Line_D, L, R, U, D );

    }

    draw_markup(&frame, width, height);
    drawTarget_2(&frame, width/2, height/2, 20);

    cvShowImage("CAMERA", &frame);

    char c = cvWaitKey(1);
        if (c == 27) {                           //  ESC button
            return 1;
        }

return 0;
}




void show_BarcodeDetector(){

    CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY); //cvCaptureFromCAM( 0 );

    assert( capture );

    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);

    double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    printf(" Video size : %.0f x %.0f\n", width, height );


    IplImage* frame=0;

    cvNamedWindow("CAMERA", CV_WINDOW_AUTOSIZE);

    int process = 0;

    while(process == 0){

        frame = cvQueryFrame( capture );
        process = barcode_show_out(*frame, width, height);
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow("CAMERA");

return;
}
