#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FaceRegion.h"
#include "GrabCut.h"

using namespace std;
using namespace cv;

const Scalar GREEN = Scalar(0, 255, 0);

void GrabCut::reset() {
	if (!mask.empty())	mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear();
	fgdPxls.clear();
	prBgdPxls.clear();
	prFgdPxls.clear();

	isInitialized = false;
	hair1State = NOT_SET;
}

void GrabCut::setImageAndWinName(const Mat& _image,	const string& _winName) {
	if (_image.empty() || _winName.empty()) return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

void GrabCut::showImage() const {
	if (image->empty() || winName->empty()) 	return;

	Mat res;
	Mat binMask;
	if (!isInitialized)	image->copyTo(res);
	else {
		getBinMask(mask, binMask);
		imwrite("Data/mask.png",binMask);
		image->copyTo(res, binMask);
		uint8_t* pixelPtr = (uint8_t*)binMask.data;
		int cn = binMask.channels();
		for(int i = 0; i < binMask.rows; i++)
		{
		    for(int j = 0; j < binMask.cols; j += cn)
		    {
		        if (pixelPtr[i*binMask.cols*cn + j*cn]>0) pixelPtr[i*binMask.cols*cn + j*cn] = 255;
		    }
		}
		GaussianBlur(binMask,binMask,Size( 15, 15 ),0,0);
	}


	if (hair1State == IN_PROCESS || hair1State == SET)
		rectangle(res, Point(hair1.x, hair1.y),
				Point(hair1.x + hair1.width, hair1.y + hair1.height), GREEN, 1);

	imshow(*winName, res);
}

void GrabCut::setRect(Rect bounding_box) {
	rect = bounding_box;
}

void GrabCut::setRectInMask() {
	assert( !mask.empty());
	mask.setTo(GC_BGD);
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}

void GrabCut::setPixelsInMask(int pixel_nature, Point p) {
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	bpxls = &bgdPxls;
	fpxls = &fgdPxls;
	bvalue = GC_BGD;
	fvalue = GC_FGD;

	if (pixel_nature == BACKGROUND) {
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);
	} else if (pixel_nature == FOREGROUND) {
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);
	} else {
		cout << "NOT WORKING" << endl;
	}
}
void GrabCut::setRectOfPixels(Rect c, int pixel_nature) {
	for (int i = c.x; i < c.x + c.width; i++) {
		for (int j = c.y; j < c.y + c.height; j++) {
			setPixelsInMask(pixel_nature, Point(i, j));
		}
	}
}

void GrabCut::imageSegment(FaceRegion* face_region) {
	Mat bgdModel, fgdModel;
	reset();
	// SET RECT HERE


	setRect(face_region->boundingRect);
	setRectInMask();
//	assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
	// SET EYES
	setRectOfPixels(face_region->eyeLeftRect, BACKGROUND);
	setRectOfPixels(face_region->eyeRightRect, BACKGROUND);
	// SET MOUTH
	setRectOfPixels(face_region->mouthRect, BACKGROUND);
	// SET NECK
	Rect neck_temp = face_region->mouthRect;
	neck_temp.height = face_region->image->height - (face_region->mouthRect.y + face_region->mouthRect.height);
	neck_temp.y = face_region->mouthRect.y + face_region->mouthRect.height;
	setRectOfPixels(neck_temp, BACKGROUND);
	// SET NOSE
	setRectOfPixels(face_region->noseRect, BACKGROUND);
	// CHEEKS AND FOREHEAD
	setRectOfPixels(face_region->skinForehead, BACKGROUND);
	setRectOfPixels(face_region->skinLeftCheek, BACKGROUND);
	setRectOfPixels(face_region->skinLeftCheek, BACKGROUND);
	//BACKGROUND
	setRectOfPixels(face_region->leftBackgroundBox, BACKGROUND);
	setRectOfPixels(face_region->rightBackgroundBox, BACKGROUND);
	// HAIR
	vector<Point>::const_iterator it;

	for (it = face_region->Hair_vector.begin(); it != face_region->Hair_vector.end(); ++it){
		fgdPxls.push_back(*it);
	}
	isInitialized = true;
	grabCut(*image, mask, face_region->boundingRect, bgdModel, fgdModel, 15, GC_INIT_WITH_MASK);
}

void getBinMask(const Mat& comMask, Mat& binMask) {
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows 	|| binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;
	cout<<binMask.data<<endl;
}
