#ifndef UTILITY_FUNCTIONS_H_
#define UTILITY_FUNCTIONS_H_

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

uchar& pixel(IplImage* canvas, int row, int col, int channel); //access pixel data
string getImageID(string filename);
void showImage(IplImage* img, CvRect roi);
double getMean(double* array, int numberOfElements);
double getSTD(double* array, int numberOfElements, double mean);
CvScalar getROIAverage(IplImage* img, int pivot_x, int pivot_y, int width,int height);
double euclideanColorDistance(CvScalar color1, CvScalar color2);

#endif /* UTILITY_FUNCTIONS_H_ */
