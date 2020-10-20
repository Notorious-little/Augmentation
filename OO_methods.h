#ifndef METHODS
#define METHODS


#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstring>
#include <cstdio>
#include <cmath>


struct Params{

    int quantil;
    int blurpower;
    int noize_range;
    double* elements;

};


class Augmentation{

    public:

    Augmentation(){}

    virtual cv::Mat makeImage(const cv::Mat &img, Params &p) = 0;

    virtual ~Augmentation(){}

};


class Autocontrast : public Augmentation{

    public :

    Autocontrast(){}
    ~Autocontrast(void){}

    cv::Mat makeImage(const cv::Mat &img, Params &p);

};


class Blur : public Augmentation{

    public :

    Blur(){}
    ~Blur(void){}

    cv::Mat makeImage(const cv::Mat &input_img, Params &p);

    private:

    void slidingNormalMatrixCompute(int blurpower, double *elements);

};


class GaussNoize : public Augmentation{

    public:

    GaussNoize(){}
    ~GaussNoize(){}

    cv::Mat makeImage( const cv::Mat &input_img, Params &p);

    private:

    double exp_rand(void);
};

#endif
