#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "gradient_map.h"
#include "methods.h"
#include "map_edje_filtration.h"


void find_circle_around_point(int x, int y, int Rad, cv::Mat &input_img){           // (x,y) - center, R - radius

    for (int R = 40; R < 120; ++R){

        int Rad_summ = 0;

        for (int a = -R-1; a <= R+1; ++a){
        for (int b = -R-1; b <= R+1; ++b){
            if( (abs((double) a - (double)R*((double)a/sqrt((double)a*(double)a+(double)b*(double)b))) < sqrt(2)) &&
                (abs((double) b - (double)R*((double)b/sqrt((double)a*(double)a+(double)b*(double)b))) < sqrt(2)) &&
                input_img.at<uchar>(x+a, y+b) > 250 )

                ++Rad_summ;

            }
        }

        if (Rad_summ > 6*M_PI*R/3)
            std::cout<<"circle in "<<x<<" "<<y<<" rad "<<R<<std::endl;

    }

return;
}

