#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "methods.h"

# define M_PI           3.14159265358979323846


int main(){
    
    cv::Mat img = cv::imread ( "./mrb.bmp" , cv::IMREAD_COLOR) ;
    
    cv::imshow( "Original", img);                     
    
    cv::Mat autoc = autoContrast(img, 5);

    cv::imshow( "Contrasted", autoc);                 

    cv::Mat blured_img = blurByGaussMatrix(img, 5);

    cv::imshow( "Blured", blured_img);

    cv::Mat contoured = localContours(img);

    cv::imshow( "Contoured", contoured);

    cv::waitKey(0);
    
    cv::imwrite("./mrbinout.bmp", autoc);             
    
return 0;
}


