
 
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utility_functions.h"
#include "Accessories.h"
#include "FaceRegion.h"

using namespace std;
using namespace cv;


Accessories::Accessories(){

}

Accessories::Accessories(FaceRegion* face_region, IplImage* glass_image) {
	image = face_region->image;

	Rect face_temp = face_region->faceRect;
	Rect eye_pair_temp = face_region->eyePairRect;
	Rect eye_right_temp = face_region->eyeRightRect;
	Rect eye_left_temp = face_region->eyeLeftRect;

	faceBox = cvRect(face_temp.x, face_temp.y, face_temp.width, face_temp.height);
	eyePair = cvRect(eye_pair_temp.x, eye_pair_temp.y, eye_pair_temp.width, eye_pair_temp.height);
	eyeLeft = cvRect(eye_left_temp.x, eye_left_temp.y, eye_left_temp.width, eye_left_temp.height);
	eyeRight = cvRect(eye_right_temp.x, eye_right_temp.y, eye_right_temp.width, eye_right_temp.height);
	glass = glass_image;

	flagLeftEyeFound = face_region->flagLeftEyeDetect;
	flagRightEyeFound = face_region->flagRightEyeDetect;

	modifiedImage = cvCreateImage(cvGetSize(image),8,3);
}

void Accessories::initializeParams(FaceRegion* face_region, IplImage* glass_image) {
	image = face_region->image;

	Rect face_temp = face_region->faceRect;
	Rect eye_pair_temp = face_region->eyePairRect;
	Rect eye_right_temp = face_region->eyeRightRect;
	Rect eye_left_temp = face_region->eyeLeftRect;

	faceBox = cvRect(face_temp.x, face_temp.y, face_temp.width, face_temp.height);
	eyePair = cvRect(eye_pair_temp.x, eye_pair_temp.y, eye_pair_temp.width, eye_pair_temp.height);
	eyeLeft = cvRect(eye_left_temp.x, eye_left_temp.y, eye_left_temp.width, eye_left_temp.height);
	eyeRight = cvRect(eye_right_temp.x, eye_right_temp.y, eye_right_temp.width, eye_right_temp.height);
	glass = glass_image;

	flagLeftEyeFound = face_region->flagLeftEyeDetect;
	flagRightEyeFound = face_region->flagRightEyeDetect;

	modifiedImage = cvCreateImage(cvGetSize(image),8,3);
}

void Accessories::put_glass(IplImage* original_img ){
	image = original_img;
	cvCopy(image,modifiedImage);
	int total, glass_height;
	CvRect eyepair_border;

	if (eyeLeft.height>eyeRight.height) glass_height = eyeLeft.height;
	else glass_height = eyeRight.height;

	if((flagLeftEyeFound==1) && (flagRightEyeFound==1)){

		eyepair_border = cvRect(eyeLeft.x-(0.05*faceBox.width), eyeLeft.y+5, eyeRight.x+eyeRight.width-eyeLeft.x+(0.1*faceBox.width), glass_height);

	}
	else if((flagLeftEyeFound==1) && (flagRightEyeFound==0)){
		eyepair_border = cvRect(eyeLeft.x, eyeLeft.y, 0.75*faceBox.width, eyeLeft.height);

	}
	else if((flagLeftEyeFound==0) && (flagRightEyeFound==1)){
		eyepair_border = cvRect(eyeRight.x+eyeRight.width-(0.75*faceBox.width), eyeRight.y, 0.75*faceBox.width, eyeRight.height);

	}
	else{
		cout<<"No eyes detected!"<<endl;
		return;

	}
	IplImage* overlay1 = cvCreateImage(cvSize(eyepair_border.width,eyepair_border.height),modifiedImage->depth,modifiedImage->nChannels);
	cvSetImageROI(modifiedImage, eyepair_border);
	cvResize(glass, overlay1);

	for(int x=0;x < overlay1->width;x++)
	{
		if(x+0>=modifiedImage->width) continue;
		for(int y=0;y < overlay1->height ;y++)
		{
			if(y+0>=modifiedImage->height) continue;
			CvScalar source = cvGet2D(modifiedImage, y, x);
			CvScalar over = cvGet2D(overlay1, y, x);
			CvScalar merged;
			CvScalar Source_mask = cvScalar(0.01,0.01,0.01,0.01);
			CvScalar over_mask = cvScalar(0.01,0.01,0.01,0.01);
			for(int i=0;i<3;i++) total = (int)over.val[i]+total;
			if (total>350)
			{
				for(int j=0;j<3;j++) merged.val[j] = (source.val[j]+over_mask.val[j]*over.val[j]);
				total = 0;
			}

			else{
				for(int k=0;k<3;k++) merged.val[k] = (Source_mask.val[k]*source.val[k]+ over.val[k]);
				total = 0;
			}

			total = 0;
			cvSet2D(modifiedImage, y, x, merged);
		}
	}
	cvResetImageROI(modifiedImage);
}
