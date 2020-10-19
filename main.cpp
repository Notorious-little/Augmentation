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
#include "methods.h"
#include "OO_methods.h"
#include "gradient_map.h"
#include "map_edje_filtration.h"
#include "find_circle_around_point.h"

# define M_PI           3.14159265358979323846



void drawTarget(IplImage* img, int x, int y, int radius)
{
    cvCircle(img,cvPoint(x, y),radius,CV_RGB(250,0,0),1,8);
    cvLine(img, cvPoint(x-radius/2, y-radius/2), cvPoint(x+radius/2, y+radius/2),CV_RGB(250,0,0),1,8);
    cvLine(img, cvPoint(x-radius/2, y+radius/2), cvPoint(x+radius/2, y-radius/2),CV_RGB(250,0,0),1,8);
}


void myMouseCallback( int event, int x, int y, int flags, void* param )
{


    switch( event ){

        case CV_EVENT_LBUTTONDOWN:

            IplImage* img = (IplImage*) param;

            printf("%d , %d \n", x, y);
            drawTarget(img, x, y, 10);
            cv::Mat input_img = cv::cvarrToMat(img);

            int h = input_img.rows;
            int w = input_img.cols;

            int Map_size = (h)*(w);
            double Map[Map_size];

            cv::Mat AC_img = autoContrast(input_img, 10);
          //  cv::Mat blured = blurByGaussMatrix(AC_img, 2);
            localGradientMap(AC_img, Map);
            cv::Mat map_img = draw_GM_contoured_img(input_img, Map, Map_size);
            cv::Mat medianed = medianFilter_8UC1(map_img);
            cv::imshow("G", medianed);


            for (int i = -2; i < 2; ++i){
                for (int j = -2; j < 2; ++j){

                    int Radius =
                            find_circle_around_point(x+i, y+j, 0, medianed);

                    if (Radius > 5){
                        cvCircle( img, cvPoint(x+i, y+j), Radius , CV_RGB(250,0,0), 1, 8);
                    }
                }
            }

            break;

    }
return;
}




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

            std::string im_name = "/Im_";
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

/*

int main() {

    IplImage* src = 0;

    cv::Mat img = cv::imread ( "./coins.bmp" , cv::IMREAD_COLOR) ;

    IplImage* image = cvCreateImage(cvSize(img.cols, img.rows), 8, 3);
    IplImage ipltemp = img;
    cvCopy(&ipltemp, image);

    int h = img.rows;
    int w = img.cols;

    src = cvCloneImage(image);

        assert( src != 0 );

        cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);

        cvSetMouseCallback( "Original", myMouseCallback, (void*) image);

        while(1){
            src = cvCloneImage( image);
            cvShowImage( "Original", src );

            char c = cvWaitKey(33);
            if (c == 27) {                 // c = 27 is Esc
                break;
            }
        }

        cvReleaseImage(&image);
        cvReleaseImage(&src);

        cvDestroyWindow("Original");



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



    int Map_size = (h+1)*(w+1);
    double Map[Map_size];
    cv::Mat blured_img = blurByGaussMatrix(img, 2);
    cv::Mat AC_img = autoContrast(img, 1);

    localGradientMap(AC_img, Map);
    cv::Mat map_img = draw_GM_contoured_img(img, Map, Map_size);
    cv::Mat medianed = medianFilter_8UC1(map_img);
    cv::imshow("GMC", medianed);

    cv::Mat dst = img.clone();
    cvtColor( img , dst , cv::COLOR_BGR2GRAY );
    cv::Canny(img, dst, 10, 200, 3);
    cv::imshow( "Canny Contoured", dst);



return 0;
}  */


