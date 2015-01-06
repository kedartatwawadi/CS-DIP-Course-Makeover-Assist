#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utility_functions.h"

using namespace std;
using namespace cv;

uchar& pixel(IplImage* canvas, int row, int col, int channel) {
	return ((uchar*) (canvas->imageData + canvas->widthStep * row))[col	* canvas->nChannels + channel];
}

string getImageID(string filename) {
	size_t found_dot = filename.find_last_of(".");
	size_t found_slash = filename.find_last_of("/");
	size_t number_of_letters = found_dot - found_slash - 1;
	string image_basename = filename.substr(found_slash + 1, number_of_letters);
	return image_basename;
}

void showImage(IplImage* img, CvRect roi) {
	const char* name = "showImage";
	cvNamedWindow(name, 1);
	IplImage* img_draw = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, img_draw, CV_HSV2RGB);
	cvRectangleR(img_draw, roi, CV_RGB(255, 0, 0), 1, 8, 0);
	cvShowImage(name, img_draw);
	cvWaitKey(0);
	cvReleaseImage(&img_draw);
}

double getMean(double* array, int numberOfElements) {
	double sum = 0;
	for (int i = 0; i < numberOfElements; i++) {
		sum = sum + array[i];
	}
	double mean = sum / numberOfElements;
	return mean;
}

double getSTD(double* array, int numberOfElements, double mean) {

	double sum = 0;
	for (int i = 0; i < numberOfElements; i++) {
		sum = sum + (array[i] - mean) * (array[i] - mean);
	}
	sum = sum / numberOfElements;
	double stdev = sqrt(sum);
	return stdev;
}

CvScalar getROIAverage(IplImage* img, int pivot_x, int pivot_y, int width, int height) {
	CvRect old_roi = cvGetImageROI(img);
	cvSetImageROI(img, cvRect(pivot_x, pivot_y, width, height));
	CvScalar c = cvAvg(img);
	cvSetImageROI(img, old_roi); // reset old roi
	return c;
}

double euclideanColorDistance(CvScalar color1, CvScalar color2) {
	double b = color1.val[0] - color2.val[0];
	double g = color1.val[1] - color2.val[1];
	double r = color1.val[2] - color2.val[2];
	double color_difference_square = r * r + g * g + b * b;
	double color_difference = sqrt(color_difference_square);
	return color_difference;
}
