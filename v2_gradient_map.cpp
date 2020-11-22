#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cmath>




bool in(int elem, int* arr, int arr_size){

    for (int i = 0; i < arr_size; ++i){
        if ( arr[i] == elem){
            return true;
        }
    }

return false;
}


cv::Mat central_connected_component (const cv::Mat &img,
                int L, int R, int U, int D, int *Square){

/* Have to cut out the region of interest by parameters L, R, U, D */
    cv::Mat output_img( img.rows , img.cols , CV_8UC1 );
    output_img = img.clone();

    int y = (int)(img.cols/2);
    int x = (int)(img.rows/2);

    if ((int)(img.at<uchar>((int)(U+(D-U)/2), (int)(L+(R-L)/2))) != 0 ){
        Square = 0;
        return img;
    }


    int *comp_map = new int[(R-L)*(D-U)]{0};
    int *conformity_tabl = new int[(R-L)*(D-U)]{0};
    int comp_count = 0;
    int tmp = 0;

    if (img.at<uchar>(U,L) == 0){
        ++comp_count;
        comp_map[0] = comp_count;
    }

    for (int j = 1; j < (R-L); ++j){

        if (img.at<uchar>(U,j+L) == 0 ){

            if ( comp_map[j-1] != 0 ){
                comp_map[j] = comp_map[j-1];
            } else {
                ++comp_count;
                comp_map[j] = comp_count;
            }
        }
    }

    for (int i = 1; i < (D-U); ++i){

        if (img.at<uchar>(i+U,L) == 0){

            if (comp_map[(i-1)*(R-L)] != 0){
                comp_map[i*(R-L)] = comp_map[(i-1)*(R-L)];
            } else {
                ++comp_count;
                comp_map[i*(R-L)] = comp_count;
            }
        }

        for (int j = 1; j < (R-L); ++j){

            if (img.at<uchar>(i+U,j+L) == 0 ){

                if ( comp_map[(i-1)*(R-L) + j] != 0 ){

                    comp_map[i*(R-L) + j] = comp_map[(i-1)*(R-L) + j];

                } else if (comp_map[i*(R-L) + j-1] != 0){

                    comp_map[i*(R-L) + j] = comp_map[i*(R-L) + j-1];

                } else if ( (comp_map[i*(R-L) + j-1] == 0) &&
                            (comp_map[(i-1)*(R-L) + j] == 0)){

                    ++comp_count;
                    comp_map[i*(R-L) + j] = comp_count;

                }

                if ( ( !(in( comp_map[(i-1)*(R-L) + j-1] , conformity_tabl, tmp))) &&
                    (comp_map[(i-1)*(R-L) + j] != 0) &&
                    (comp_map[i*(R-L) + j - 1] != 0) &&
                    (comp_map[(i-1)*(R-L) + j] != comp_map[i*(R-L) + j - 1] )) {

                    conformity_tabl[tmp] = comp_map[(i-1)*(R-L) + j];
                    ++tmp;
                    conformity_tabl[tmp] = comp_map[ i*(R-L) + j - 1];
                    ++tmp;

                }

            }
        }
    }


/* Here we got a map with a center component */

    int central_comp_number = comp_map[(int)(((D-U)/2)*(R-L) + (R-L)/2) ];  //  Original value of central pixel
    assert ( central_comp_number != 0);

    if (central_comp_number == 0) {
        return img;
    }

    int central_num_t = -1;
    int *conformity_tabl_orig = new int[(R-L)*(D-U)]{0};

    for (int i = 0; i < tmp; ++i){
        conformity_tabl_orig[i] =
                conformity_tabl[i];
    }

    for (int i = 0; i < tmp; i += 2){
        if (conformity_tabl[i] == central_comp_number){
            central_num_t = i;                              // Position of center pixel in eq. table
        }
    }
    if (central_num_t == -1){
        *Square = -2;
        return img;
    }

    for (int k = 0; k < (int)(tmp/2); ++k){
        for(int i = 0; i < tmp; i += 2){

            if(conformity_tabl[i] == conformity_tabl[2*k +1]){

                conformity_tabl[i] = conformity_tabl[2*k];

            }
        }
    }


    int new_cent_num_table = conformity_tabl[central_num_t];
    int j = 0;
    int* cent_accum = new int[(D-U)*(R-L)]{0};

    for (int i = 0; i < tmp; ++i){
        if ( conformity_tabl[i] == new_cent_num_table ) {
            cent_accum[j] = conformity_tabl_orig[i];
            ++j;
        }
    }

    delete conformity_tabl;
    delete conformity_tabl_orig;

    int* center_eq = new int[(R-L)*(D-U)]{0};
    int cent_eq_size = 1;
    int swap = 0;


    for (int k = 0; k < j-1; ++k){
        for (int m = 0; m < j-1; ++m){
            if (cent_accum[m] < cent_accum[m+1]){
                swap = cent_accum[m];
                cent_accum[m] = cent_accum[m+1];
                cent_accum[m+1] = swap;
            }
        }
    }

    center_eq[0] = cent_accum[0];

    for (int k = 1; k < j; ++k){
        if ( cent_accum[k] != cent_accum[k-1]){
            center_eq[cent_eq_size] = cent_accum[k];
            ++cent_eq_size;
        }
    }

    delete cent_accum;

    for ( int i = 0; i < (R-L); ++i){
        if (in(comp_map[i], center_eq, cent_eq_size)){
            *Square = -1;
            return img;
        }
    }

    for (int i = 0 ; i < (D-U); ++i){
        for (int j = 0; j < R-L; ++j){

            if( in(comp_map[i*(R-L) + j], center_eq, cent_eq_size) ){

                output_img.at<uchar>( U + i ,
                                      L + j) = 0;
                ++ *Square;
            } else {
                output_img.at<uchar>( U + i ,
                                      L + j) = 255;
            }
        }
    }

    delete comp_map;
    delete center_eq;

return output_img;
}





// Здесь составим "карту градиентов" Map[h*w] изображения скользящим 7х7 окном
// Предварительно оно обработано фильтрами Blur, (0-quantil linear) Autocontrast
// Будем считать, что в точках "карты" такие обозначения :
// 0 - градиент не прошел проверку на локальный максимум в собственной 3-окрестности
// 1 - градиент вертикален
// 2 - градиент горизонтален
// 3 - градиент параллелен "главной диагонали"
// 4 - градиент параллелен "побочной диагонали"


void wide_Window_GradientMap(const cv::Mat &input_img, double* Map, const int RGB){

    int h = input_img.rows;
    int w = input_img.cols;

    cv::Mat img = input_img.clone();

    double P1 = 2;
    double P2 = 1;

    int x[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};
    int y[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    double k[16];
    
    for (int i = 0; i < 16; ++i){
        k[i] = sqrt ( (x[i]-x[i+8])*(x[i]-x[i+8])
                    + (y[i]-y[i+8])*(y[i]-y[i+8]) );
    } 

    double av_grad = 1;

    for(int i = 3; i < h-3; ++i){
        for(int j = 3; j < w-3; ++j){

            double grad[8];
            double grad_max=0, grad_min=256;
            int num_max = 0, num_min = 0;

            for (int s = 0; s < 8; ++s){
                grad[s] = abs ( img.at<cv::Vec3b>(i+x[s], j+y[s])[RGB]
                              - img.at<cv::Vec3b>(i+x[s+8], j+y[s+8])[RGB] );
            }

            double grad_av[4] = { (grad[0]/k[0] + grad[1]/k[1] + grad[2]/k[2])/3,
                                  (grad[2]/k[2] + grad[3]/k[3] + grad[4]/k[4])/3,
                                  (grad[4]/k[4] + grad[5]/k[5] + grad[6]/k[6])/3,
                                  (grad[6]/k[6] + grad[7]/k[7] + grad[8]/k[8])/3};

            for (int i = 0; i < 4; ++i){

                if(grad_av[i] > grad_max){
                    grad_max = grad_av[i];
                    num_max = i;
                }

                if(grad_av[i] < grad_min){
                    grad_min = grad_av[i];
                    num_min = i;
                }
            }

            if ( ( grad_max > P2 * av_grad ) &&
                 (grad_max > P1 * grad_min) &&
                 ( ( (abs(num_max - num_min)) % 2) == 0 ) ){

                switch(num_max){
                    case 0:

                        Map[i*w+j] = 2;
                        break;

                    case 1:

                        Map[i*w+j] = 4;
                        break;

                    case 2:

                        Map[i*w+j] = 1;
                        break;

                    case 3:

                        Map[i*w+j] = 3;
                        break;

                }

            } else {
                Map[i*w + j] = 0;
            }


        }
    }

return;
}




void wide_Window_map(const cv::Mat &input_img, double* Result_Map, int detnum){

    int h = input_img.rows;
    int w = input_img.cols;
    cv::Mat img = input_img.clone();

    double* Map_R = new double[h*w] ;
    double* Map_G = new double[h*w];
    double* Map_B = new double[h*w];
    double Result_val = 0;

    wide_Window_GradientMap(img, Map_R, 0);
    wide_Window_GradientMap(img, Map_G, 1);
    wide_Window_GradientMap(img, Map_B, 2);

    for(int i = 0; i < h*w; ++i){

        if (Map_R[i] = Map_B[i]){
            ++Result_val;
            Result_Map[i] = Map_R[i];
        }

        if (Map_R[i] = Map_G[i]){
            ++Result_val;
            Result_Map[i] = Map_R[i];
        }

        if (Map_G[i] = Map_B[i]){
            ++Result_val;
            Result_Map[i] = Map_G[i];
        }

        if (Result_val < detnum){
            Result_Map[i] = 0;
        }

        Result_val = 0;

    }


    delete Map_R;
    delete Map_B;
    delete Map_G;

return;
}

