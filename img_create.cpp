#include <cassert>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "methods.h"
#include "OO_methods.h"
#include "gradient_contour.h"

# define M_PI           3.14159265358979323846



int main(int argc, char *argv[]){

    if(argc != 4){
        std::cout << " Please enter ./executable " <<
                     "<Input folder-file path> <Output folder path> "
                     "<Number of Augmentations>" << std::endl;
        return 1;
    }

    Params p{5, 3, 10};                     // quantil = 5; blurpower = 3; noize_range = 10;
    std::vector<Augmentation*> defect = {new Blur(), new GaussNoize(), new Autocontrast()};

    int Num = 0;
    int n = std::atoi(argv[3]);
    std::string im_path;

    std::ifstream file;
    file.open(argv[1]);

    if ( !(file.is_open()) ) {
        std::cout<<"File was not opened"<<std::endl;
        return 2;
    };

    cv::Mat newImage;

    while (std::getline(file, im_path)){

        cv::Mat img = cv::imread ( im_path , cv::IMREAD_COLOR) ;
        std::srand( (int)time(NULL) );
        int times = rand() % n;

        for(int k = 0; k < times; ++k){

            std::srand( (int)time(NULL) );
            p.quantil = ( 2*abs( (int) std::rand() ) ) %10;
            p.blurpower = ( abs( (int) std::rand() ) ) % 5;
            p.noize_range = ( abs( (int) std::rand() ) ) % 10;

            std::srand( (int)time(NULL) );
            int a = std::rand();

            newImage = defect[ (a % 3) ]->makeImage(img, p);
        }

        std::string im_name = "/Im_";
        im_name += std::to_string(Num);
        im_name += ".bmp";

        std::string im_out_path = argv[2] + im_name;
        ++Num;
        cv::imwrite(im_out_path, newImage);

        std::cout<<im_out_path<<std::endl;

        std::cout<<"MADE"<<std::endl;

    }

    for(int i = 0; i < 3; ++i){
        delete defect.at(i);
    }

    file.close();

return 0;
}

/*
int main(){

    cv::Mat img = cv::imread ( "./boeing.bmp" , cv::IMREAD_COLOR) ;

    cv::imshow( "Original", img);

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
    cv::imwrite("./mrbinout.bmp", autoc);  


    Params p{5, 3, 10};                     // quantil = 5; blurpower = 3; noize_range = 10;

    std::vector<Augmentation*> composition = {new Blur(), new GaussNoize(), new Autocontrast()};

    std::srand( (int)time(NULL) );

    int i = std::rand();

    cv::Mat newImage;

    newImage = composition[ (i % 3) ]->makeImage(img, p);

    int j = std::rand();
    int k = std::rand();

    if (j % 4 != 3)
        newImage = composition[ (j % 4) ]->makeImage(newImage, p);

    if (k % 4 != 3)
        newImage = composition[ (k % 4) ]->makeImage(newImage, p); 

    cv::Mat AC_img = autoContrast(img, 1);
    cv::Mat blured_img = blurByGaussMatrix(img, 2);

    cv::Mat contoured = localContours(blured_img);
    cv::Mat cont_med = medianFilter_8UC1(contoured );
    cv::Mat cont_med2 = medianFilter_8UC1(cont_med );


    cv::imshow( "Contoured", cont_med2);
    cv::imshow( "Orig", img);

    cv::waitKey(0);  

return 0;
}     */
