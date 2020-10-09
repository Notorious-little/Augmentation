#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "methods.h"
#include "OO_methods.h"

# define M_PI           3.14159265358979323846

int main(){
    
    cv::Mat img = cv::imread ( "./mrb.bmp" , cv::IMREAD_COLOR) ;

    cv::imshow( "Original", img);

    /*
    cv::Mat autoc = autoContrast(img, 5);

    cv::imshow( "Contrasted", autoc);                 

    cv::Mat blured_img = blurByGaussMatrix(img, 5);

    cv::imshow( "Blured", blured_img);

    cv::Mat contoured = localContours(img);

    cv::imshow( "Contoured", contoured);

    cv::Mat noized_image = gaussNoize( img, 15);

    cv::imshow( "Noized", noized_image);

    cv::Mat SaltPaperNoized_image = salt_paperNoize( img, 8083647);

    cv::imshow( "SL-Noized", SaltPaperNoized_image);

    cv::waitKey(0);
    
    cv::imwrite("./mrbinout.bmp", autoc); */


    Params p;
    p.quantil = 5;
    p.blurpower = 3;
    p.noize_range = 10;

    std::vector<Augmentation*> composition = {new Blur(), new GaussNoize(), new Autocontrast()};

    std::srand( time(NULL) );

    int i = std::rand();
    int j = rand();
    int k = rand();

    cv::Mat newImage;


    newImage = composition[ (int)(i % 3) ]->makeImage(img, p);

    if (j % 4 != 3)
        newImage = composition[ (int)(j % 4) ]->makeImage(newImage, p);

    if (k % 4 != 3)
        newImage = composition[ (int)(k % 4) ]->makeImage(newImage, p);

    std::cout<<" random numbers "<<i<<" "<<j<<" "<<k<<" "<<std::endl;

    cv::imshow( "RANDOM AUG", newImage);

    cv::waitKey(0);

return 0;
}

