#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utility_functions.h"
#include "HairRegion.h"

using namespace std;
using namespace cv;

int hue_value;
int saturation_value;

HairRegion::HairRegion() {

}

HairRegion::HairRegion(IplImage* original_image,IplImage* mask_image) {
	originalImage = original_image;
	mask = mask_image;
	modifiedImage = cvCreateImage(cvGetSize(originalImage), 8, 3);
}

void HairRegion::setImageAndMask(IplImage* original_image,IplImage* mask_image) {
	originalImage = cvCreateImage(cvGetSize(original_image), 8, 3);
	mask = cvCreateImage(cvGetSize(mask_image), 8, 1);
	modifiedImage = cvCreateImage(cvGetSize(originalImage), 8, 3);

	cvCopy(original_image,originalImage);
	cvCopy(mask_image,mask);
}

void HairRegion::scaleMask() {
	for (int i = 0; i<mask->width; i++) {
		for (int j = 0; j < mask->height; j++) {
			if (pixel(mask,j,i,0) > 0) pixel(mask, j, i, 0) = 255;
		}
	}
	cvErode(mask,mask,NULL,1);
	cvDilate(mask,mask,NULL,1);
	cvSmooth(mask,mask,CV_GAUSSIAN);
	cvNamedWindow("mask",1);
	cvMoveWindow("mask",100,500);
	cvShowImage("mask",mask);
	while(cvWaitKey(10) > 0);
}

void HairRegion::setRequiredHairHueSaturation(unsigned char hue, unsigned char saturation) {
	requiredHairHue = hue;
	requiredHairSaturation = saturation;
}

void HairRegion::changeHueSaturation() {
	IplImage* original = cvCreateImage(cvGetSize(originalImage), 8, 3);
	cvCopy(originalImage,original);

	IplImage* temp_rgb = cvCreateImage(cvGetSize(originalImage), 8, 3);
	cvCopy(originalImage,temp_rgb);

	IplImage* hair_mask = cvCreateImage(cvGetSize(originalImage), 8, 1);
	cvCopy(mask,hair_mask);

	IplImage* temp_hsv = cvCreateImage(cvGetSize(originalImage), 8, 3);
	//IplImage* temp_hist_equalized = cvCreateImage(cvGetSize(originalImage), 8, 1);
	IplImage* hue = cvCreateImage(cvGetSize(originalImage), 8, 1);
	IplImage* sat = cvCreateImage(cvGetSize(originalImage), 8, 1);
	IplImage* val = cvCreateImage(cvGetSize(originalImage), 8, 1);


	//IplImage* temp = cvCreateImage(cvSize(hair_hsv->width, hair_hsv->height), hair_hsv->depth, 1 );
	cvCvtColor(temp_rgb, temp_hsv, CV_BGR2HSV);
	cvSplit(temp_hsv, hue, sat, val, 0);
	//cvThreshold(hue, hue, 100, 120, CV_THRESH_TOZERO);
	for (int i=0; i<hue->width; i++) {
		for(int j=0; j<hue->height; j++) {
			if (pixel(hair_mask,j,i,0) > 0) {
				pixel(hue,j, i, 0) = requiredHairHue;//(pixel(hue,j, i, 0)+requiredHairHue)%256;
				pixel(sat,j, i, 0) = requiredHairSaturation;
			}
		}
	}
//	cvEqualizeHist(val,temp_hist_equalized);
//	cvNamedWindow("hist-eq");
//	cvShowImage("hist-eq",temp_hist_equalized);
//	cvWaitKey(0);
	cvMerge(hue, sat, val, 0, temp_hsv);

	cvCvtColor(temp_hsv,temp_rgb, CV_HSV2BGR);

	//cvCvtColor(temp_rgb,temp_hist_equalized,CV_BGR2GRAY);

	//cvCvtColor(temp_hist_equalized,temp_rgb,CV_GRAY2BGR);
	double alpha;
	for (int i=0; i<original->width; i++){
		for(int j=0; j<original->height; j++) {
			for (int k=0;k<3;k++){
				alpha = (double) pixel(hair_mask,j,i,0)/255;
				pixel(modifiedImage,j,i,k) = alpha*pixel(temp_rgb,j,i,k) + (1-alpha)*pixel(original,j,i,k);
			}
		}
	}

//	cvSmooth(final,final,CV_GAUSSIAN);
//	cvNamedWindow("final",1);
//	cvShowImage("final", modifiedImage);
//	cvWaitKey(0);

	cvReleaseImage(&original);
	cvReleaseImage(&hair_mask);
	cvReleaseImage(&temp_rgb);
	cvReleaseImage(&temp_hsv);
	cvReleaseImage(&hue);
	cvReleaseImage(&sat);
	cvReleaseImage(&val);
	//cvReleaseImage(&temp_hist_equalized);
}
