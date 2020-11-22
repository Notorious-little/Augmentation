# define M_PI           3.14159265358979323846

#include <cassert>
#include <fstream>
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
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include "methods.h"
#include "OO_methods.h"
#include "gradient_map.h"
#include "map_filtrations.h"
#include "find_circle_around_point.h"
#include "v2_gradient_map.h"
#include "support.h"
#include "barcode.h"



int main(int argc, char* argv[])
{
   // cv::Mat img = cv::imread( "./coins3.bmp" , cv::IMREAD_COLOR) ;

   // show_OriginalImage(img);
    show_BarcodeDetector();

}


/*
int main(int argc, char *argv[]){

    if(argc != 4){
        std::cout << " Please enter ./executable " <<
                     "<Input folder-file path> <Output folder path> "
                     "<Number of Augmentations>" << std::endl;
        return 1;
    }

    Params p{5, 3, 10};                     // quantil = 5; blurpower = 3; noize_range = 10;

    std::vector<Augmentation*> defect = { new GaussNoize(), new Blur(), new Autocontrast()};

    int Num = 0;
    int n = std::atoi(argv[3]);
    std::string im_path;

    std::ifstream file;
    file.open(argv[1]);

    if ( !(file.is_open()) ) {
        std::cout<<"File was not opened"<<std::endl;
        return 2;
    };

    std::default_random_engine rand_gen;
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    rand_gen.seed(seed);
    std::uniform_int_distribution<int> distribution(0, defect.size() );

    while (std::getline(file, im_path)){

        std::cout<<im_path<<"     ";
        cv::Mat img = cv::imread ( im_path , cv::IMREAD_COLOR) ;

        for(int m = 0; m < n; ++m){

            std::shuffle(std::begin(defect), std::end(defect), rand_gen);

            int times = distribution(rand_gen);

            cv::Mat newImage = img.clone();

            for(int k = 0; k < times; ++k){
                assert (k < defect.size() );
                newImage = defect[k]->makeImage(newImage, p);
            }

            std::string iимеющий данных и состоящий в основном из чисто виртуальных функций. Такое решение позволяеm_name = "/Im_";
            im_name += std::to_string(Num);
            im_name += "_";
            im_name += std::to_string(m);
            im_name += ".bmp";

            std::string im_out_path = argv[2] + im_name;
            cv::imwrite(im_out_path, newImage);
            std::cout<<im_out_path<<std::endl;

            std::cout<<"MADE"<<std::endl;
        }

        ++Num;
    }

    for(int i = 0; i < defect.size(); ++i){
        delete defect.at(i);
    }

    file.close();

return 0;
}

*/

