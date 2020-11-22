#include <QApplication>
#include <QDebug>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
 
    Mat input = imread("/home/pavelk/Projects/OpenCVwrapPerspective/sheet.jpg");

    Point2f inputQuad[4];
    inputQuad[0] = Point2f( 20, 340 );
    inputQuad[1] = Point2f( 860,110 );
    inputQuad[2] = Point2f( 1160, 650 );
    inputQuad[3] = Point2f( 200, 950 );

    Point2f outputQuad[4];
    outputQuad[0] = Point2f( 0, 0 );
    outputQuad[1] = Point2f( 870, 0 );
    outputQuad[2] = Point2f( 870, 620 );
    outputQuad[3] = Point2f( 0, 620 );

    Mat M = getPerspectiveTransform( inputQuad, outputQuad );

    vector<Point2f> inputCorners(4);
    inputCorners[0]=Point2f(0, 0);
    inputCorners[1]=Point2f(input.cols, 0);
    inputCorners[2]=Point2f(0, input.rows);
    inputCorners[3]=Point2f(input.cols, input.rows);

    vector<Point2f> outputCorners(4);
    perspectiveTransform(inputCorners, outputCorners, M);

    Rect br= boundingRect(outputCorners);

    for(int i=0; i<4; i++) {
        outputQuad[i]+=Point2f(-br.x,-br.y);
    }

    M = getPerspectiveTransform( inputQuad, outputQuad );

    warpPerspective(input, output, M, br.size());

    resize(input, input, Size(1000,1000));
    imshow("Input", input);
    resize(output, output, Size(1000,1000));
    imshow("Output2", output);

    waitKey(5000);

    return app.exec();
}
