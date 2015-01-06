#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FaceRegion.h"
#include "FaceFeature.h"
#include "utility_functions.h"

using namespace std;
using namespace cv;

CvMemStorage *storage;
//const char *file_frontalface = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml";
//const char *file_left_eye = "/usr/share/opencv/haarcascades/haarcascade_mcs_lefteye.xml";
//const char *file_right_eye = "/usr/share/opencv/haarcascades/haarcascade_mcs_righteye.xml";
//const char *file_nose = "/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml";
//const char *file_mouth = "/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml";

const char *file_frontalface = "Other/haarcascades/haarcascade_frontalface_alt2.xml";
const char *file_left_eye = "Other/haarcascades/haarcascade_mcs_lefteye.xml";
const char *file_right_eye = "Other/haarcascades/haarcascade_mcs_righteye.xml";
const char *file_nose = "Other/haarcascades/haarcascade_mcs_nose.xml";
const char *file_mouth = "Other/haarcascades/haarcascade_mcs_mouth.xml";

FaceFeature face(file_frontalface,1.1,2,CV_HAAR_DO_CANNY_PRUNING, cvSize(50,50));
FaceFeature left_eye(file_left_eye, 1.15, 3,0, cvSize(18,12));
FaceFeature right_eye(file_right_eye, 1.15, 3,0,cvSize(18,12));
FaceFeature nose(file_nose, 1.2, 3,0,cvSize(18,12));
FaceFeature mouth(file_mouth, 1.12, 3 ,0, cvSize(25,15));

//constructor
FaceRegion::FaceRegion(IplImage* img) {
	image = img;

	displayImage = cvCreateImage(cvGetSize(image), 8, 3);
	cvCopy(image, displayImage);

	floodfillImage = cvCreateImage(cvGetSize(image), 8, 3);
	cvCopy(image, floodfillImage);
}

//functions
void FaceRegion::findAllFaceFeatures() {
	storage = cvCreateMemStorage(0);
	assert(storage);

	findFace();
	findLeftEye();
	findRightEye();
	findNose();
	findMouth();
	findForehead();
	findLeftCheek();
	findRightCheek();
	findAvgSkinColors();
	findGrabcutBoundingRect();
	findAvgHairColor();
	tagHairPixels();
	findLeftBackgroundBox();
	findRightBackgroundBox();

	cvClearMemStorage(storage);
}

void FaceRegion::findFace() {
	Rect* rectptr = &faceRect;
	CvRect face_roi = cvGetImageROI(image);
	flagFaceDetect = detectFeature(face_roi,face,rectptr);
}

void FaceRegion::findLeftEye() {
	Rect* rectptr = &eyeLeftRect;
	CvRect left_eye_roi = cvRect(faceRect.x, faceRect.y + (faceRect.height / 4.5), faceRect.width / 2, faceRect.height / 2.5);
	flagLeftEyeDetect = detectFeature(left_eye_roi,left_eye,rectptr);
}

void FaceRegion::findRightEye() {
	Rect* rectptr = &eyeRightRect;
	CvRect right_eye_roi = cvRect(faceRect.x + faceRect.width / 2, faceRect.y + (faceRect.height / 4.5), faceRect.width / 2, faceRect.height / 2.5);
	flagRightEyeDetect = detectFeature(right_eye_roi,right_eye,rectptr);
}

void FaceRegion::findNose() {
	Rect* rectptr = &noseRect;
	CvRect nose_roi = cvRect(faceRect.x, faceRect.y + (faceRect.height / 2.5), faceRect.width, faceRect.height / 2);
	flagNoseDetect = detectFeature(nose_roi,nose,rectptr);
}

void FaceRegion::findMouth() {
	Rect* rectptr = &mouthRect;
	CvRect mouth_roi = cvRect(faceRect.x, faceRect.y + (faceRect.height / 1.5), faceRect.width, faceRect.height / 3);
	flagMouthDetect = detectFeature(mouth_roi,mouth,rectptr);
}

void FaceRegion::findForehead() {
	if (flagLeftEyeDetect && flagRightEyeDetect) {
		int xc_lefteye_temp = eyeLeftRect.x + eyeLeftRect.width/2;
		int xc_righteye_temp = eyeRightRect.x + eyeRightRect.width/2;

		skinForehead.width = (xc_righteye_temp - xc_lefteye_temp) / 2;
		skinForehead.height = (-faceRect.y + MIN(eyeLeftRect.y, eyeRightRect.y)) / 3;
		skinForehead.x = (xc_lefteye_temp + xc_righteye_temp)/2 - skinForehead.width / 2;
		skinForehead.y = (faceRect.y + 2*MIN(eyeLeftRect.y, eyeRightRect.y))/3 - skinForehead.height/3;
	}

	else if (!flagLeftEyeDetect && flagRightEyeDetect) {
		skinForehead.width = faceRect.width / 4;
		skinForehead.height = (eyeRightRect.y - faceRect.y)/3;
		skinForehead.x = faceRect.x + 3 * faceRect.width / 8;
		skinForehead.y = (faceRect.y + eyeRightRect.y) / 2;
	}

	else if (flagLeftEyeDetect && !flagRightEyeDetect) {
		skinForehead.width = faceRect.width / 4;
		skinForehead.height = (eyeLeftRect.y - faceRect.y)/3;
		skinForehead.x = faceRect.x + 3 * faceRect.width / 8;
		skinForehead.y = (faceRect.y + eyeLeftRect.y) / 2;
	}

	if (flagLeftEyeDetect || flagRightEyeDetect)
		cvRectangle(displayImage, cvPoint(skinForehead.x, skinForehead.y),	cvPoint(skinForehead.x + skinForehead.width, skinForehead.y + skinForehead.height),	CV_RGB(0, 0,255), 1, 8, 0);
}

void FaceRegion::findLeftCheek() {
	if (flagLeftEyeDetect && flagRightEyeDetect) {
		int xc_lefteye_temp = eyeLeftRect.x + eyeLeftRect.width/2;
		int yc_lefteye_temp = eyeLeftRect.y + eyeLeftRect.height/2;

		skinLeftCheek.width = eyeLeftRect.width * 2 / 3;
		skinLeftCheek.height = eyeLeftRect.height * 2 / 3;
		skinLeftCheek.x = (eyeLeftRect.x + 3 * xc_lefteye_temp) / 4 - skinLeftCheek.width / 2;
		skinLeftCheek.y = (3*(eyeLeftRect.y + eyeLeftRect.height) - yc_lefteye_temp) / 2;
	}

	else if (!flagLeftEyeDetect && flagRightEyeDetect) {
		int yc_righteye_temp = eyeRightRect.y + eyeRightRect.height/2;

		skinLeftCheek.width = eyeRightRect.width*2/3;
		skinLeftCheek.height = eyeRightRect.height*2/3;
		skinLeftCheek.x = 2*faceRect.x + faceRect.width - (eyeRightRect.x + eyeRightRect.width);
		skinLeftCheek.y = (3*(eyeRightRect.y+eyeRightRect.height) - yc_righteye_temp) / 2;
	}

	else if (flagLeftEyeDetect && !flagRightEyeDetect) {
		int xc_lefteye_temp = eyeLeftRect.x + eyeLeftRect.width/2;
		int yc_lefteye_temp = eyeLeftRect.y + eyeLeftRect.height/2;

		skinLeftCheek.width = eyeLeftRect.width*2/3;
		skinLeftCheek.height = eyeLeftRect.height*2/3;
		skinLeftCheek.x = (eyeLeftRect.x + 3*xc_lefteye_temp)/4 - skinLeftCheek.width/2;
		skinLeftCheek.y = (3*(eyeLeftRect.y+eyeLeftRect.height) - yc_lefteye_temp)/2;
	}

	if (flagLeftEyeDetect || flagRightEyeDetect)
		cvRectangle(displayImage, cvPoint(skinLeftCheek.x, skinLeftCheek.y),	cvPoint(skinLeftCheek.x + skinLeftCheek.width, skinLeftCheek.y + skinLeftCheek.height),	CV_RGB(0, 0,255), 1, 8, 0);
}


void FaceRegion::findRightCheek() {
	if (flagLeftEyeDetect && flagRightEyeDetect) {
		int xc_righteye_temp = eyeRightRect.x + eyeRightRect.width/2;
		int yc_righteye_temp = eyeRightRect.y + eyeRightRect.height/2;

		skinRightCheek.width = eyeRightRect.width * 2 / 3;
		skinRightCheek.height = eyeRightRect.height * 2 / 3;
		skinRightCheek.x = (3 * xc_righteye_temp + (eyeRightRect.x + eyeRightRect.width)) / 4 - skinRightCheek.width / 2;
		skinRightCheek.y = (3 * (eyeRightRect.y + eyeRightRect.height) - yc_righteye_temp) / 2;
	}

	else if (!flagLeftEyeDetect && flagRightEyeDetect) {
		int xc_righteye_temp = eyeRightRect.x + eyeRightRect.width/2;

		skinRightCheek.width = skinLeftCheek.width;
		skinRightCheek.height = skinLeftCheek.height;
		skinRightCheek.y = skinLeftCheek.y;
		skinRightCheek.x = (3*xc_righteye_temp + (eyeRightRect.x+eyeRightRect.width))/4 - skinRightCheek.width / 2;
	}

	else if (flagLeftEyeDetect && !flagRightEyeDetect) {
		skinRightCheek.width = skinLeftCheek.width;
		skinRightCheek.height = skinLeftCheek.height;
		skinRightCheek.y = skinLeftCheek.y;
		skinRightCheek.x = 2*faceRect.x + faceRect.width - (eyeLeftRect.x + skinRightCheek.width);
	}

	if (flagLeftEyeDetect || flagRightEyeDetect)
		cvRectangle(displayImage, cvPoint(skinRightCheek.x, skinRightCheek.y),	cvPoint(skinRightCheek.x + skinRightCheek.width, skinRightCheek.y + skinRightCheek.height),	CV_RGB(0, 0,255), 1, 8, 0);
}

void FaceRegion::findAvgSkinColors() {
	avgForeheadSkinColor = getROIAverage(image, skinForehead.x, skinForehead.y, skinForehead.width, skinForehead.height);
	avgLeftCheekSkinColor = getROIAverage(image, skinLeftCheek.x, skinLeftCheek.y, skinLeftCheek.width, skinLeftCheek.height);
	avgRightCheekSkinColor = getROIAverage(image, skinRightCheek.x, skinRightCheek.y, skinRightCheek.width, skinRightCheek.height);
}

void FaceRegion::findGrabcutBoundingRect() {
//	boundingRect.x = 10;
//	boundingRect.y = 10;
//	boundingRect.width = image->width-20;
//	boundingRect.height = image->height-20;

	boundingRect.x = 0.25 * (faceRect.x);
	boundingRect.y = 0.05*(faceRect.y);
	boundingRect.height = 0.25 * (3*image->height + faceRect.height);
	boundingRect.width = 0.25 * (3*image->width + faceRect.width);
	cvRectangle(displayImage,cvPoint(boundingRect.x,boundingRect.y),cvPoint(boundingRect.x+boundingRect.width,boundingRect.y+boundingRect.height),CV_RGB(255,0,255),1,8,0);
//	cout<<"found grabcut bounding bounding box"<<endl;
}

void FaceRegion::findAvgHairColor() {
	//start box to detect hair
	CvRect scan_box_temp;
	scan_box_temp.width = skinForehead.width;
	scan_box_temp.height = skinForehead.height / 2;
	scan_box_temp.x = skinForehead.x;
	scan_box_temp.y = faceRect.y;
	cvRectangleR(displayImage, scan_box_temp, CV_RGB(255,255,255));

	flagHairSeedDetect = NOT_DETECTED;
	double max_box_color_difference_threshold = 60;
	double max_pixel_color_difference_threshold = 70;
	double hair_detect_threshold = 0.9;

	while (scan_box_temp.y > scan_box_temp.height) {
		if (!flagHairSeedDetect) {
			CvScalar scan_box_avg = getROIAverage(image, scan_box_temp.x, scan_box_temp.y,	scan_box_temp.width, scan_box_temp.height);
			double difference = euclideanColorDistance(scan_box_avg, avgForeheadSkinColor);
			if (difference < max_box_color_difference_threshold) scan_box_temp.y -= scan_box_temp.height;
			else {
				int num_pixels_color_diff = 0;
				CvScalar current_pixel_color;
				for (int i = scan_box_temp.x;	i < scan_box_temp.x + scan_box_temp.width; i++) {
					for (int j = scan_box_temp.y;	j < scan_box_temp.y + scan_box_temp.height; j++) {
						current_pixel_color = cvScalar((double) pixel(image, j, i, 0),(double) pixel(image, j, i, 1),(double) pixel(image, j, i, 2));
						if (euclideanColorDistance(current_pixel_color,	avgForeheadSkinColor) > max_pixel_color_difference_threshold) num_pixels_color_diff++;
					}
				}
				if (num_pixels_color_diff > (hair_detect_threshold * scan_box_temp.width * scan_box_temp.height)) {
					flagHairSeedDetect = DETECTED;
					avgHairColor = getROIAverage(image, scan_box_temp.x,scan_box_temp.y - 3*scan_box_temp.height,scan_box_temp.width, 2*scan_box_temp.height);
					cvRectangleR(displayImage,cvRect(scan_box_temp.x,scan_box_temp.y - 3*scan_box_temp.height,scan_box_temp.width, 2*scan_box_temp.height),CV_RGB(255,0,0));
					hairFloodfillSeed = scan_box_temp;
					hairFloodfillSeed.y -= 2*hairFloodfillSeed.height;
					hairFloodfillSeed.width = hairFloodfillSeed.width/2;
					hairFloodfillSeed.x += hairFloodfillSeed.width/2;
					cvRectangleR(displayImage,hairFloodfillSeed, CV_RGB(255,255,255));
				}
				else scan_box_temp.y -= scan_box_temp.height;
			}
		}
		else if (flagHairSeedDetect) {
			break;
		}
	}
}

void FaceRegion::tagHairPixels() {
	if (flagHairSeedDetect){
		hairFloodfillBoundingBox.width = faceRect.width;
		hairFloodfillBoundingBox.x = faceRect.x;
		hairFloodfillBoundingBox.y = faceRect.y/4;
		determineHairFloodfillBoundingBoxHeight();

		cvRectangleR(displayImage,hairFloodfillBoundingBox,CV_RGB(0,255,255),1,8,0);
		floodfill(hairFloodfillSeed,avgHairColor,CV_RGB(0,0,255),hairFloodfillBoundingBox,&Hair_vector);
//		cvNamedWindow("flood_fill_demo", 1);
//		cvMoveWindow("flood_fill_demo",100,50);
//		cvShowImage("flood_fill_demo", floodfillImage);
//		cvWaitKey(0);
//		cout<<"finished flood fill"<<endl;
	}
}

void FaceRegion::findLeftBackgroundBox() {
	leftBackgroundBox.x = 0;
	leftBackgroundBox.y = 0;
	leftBackgroundBox.width = 0.25 * (faceRect.x);
	leftBackgroundBox.height = 0.25 * (faceRect.y);
	cvRectangleR(displayImage,cvRect(leftBackgroundBox.x,leftBackgroundBox.y,leftBackgroundBox.width,leftBackgroundBox.height),CV_RGB(120,120,0));
}

void FaceRegion::findRightBackgroundBox() {
	rightBackgroundBox.width = 0.25*(image->width - (faceRect.x + faceRect.width));
	rightBackgroundBox.height = 0.25 * (faceRect.y);
	rightBackgroundBox.x = image->width -1 - rightBackgroundBox.width;
	rightBackgroundBox.y = 0;
	cvRectangleR(displayImage,cvRect(rightBackgroundBox.x,rightBackgroundBox.y,rightBackgroundBox.width,rightBackgroundBox.height),CV_RGB(120,120,0));
}

int FaceRegion::detectFeature(CvRect featureROI, FaceFeature face_feature, Rect* feature_box) {
	cvSetImageROI(image, featureROI);
	CvSeq* feature;

	feature = cvHaarDetectObjects(image,
			face_feature.featureCascade,
			storage,
			face_feature.haarScaleFactor,
			face_feature.haarMinNeighbours,
			face_feature.haarFlags,
			face_feature.minFeatureSize);

	cvResetImageROI(image);
	CvRect* r;
	int index_max_area;
	int x1, x2, y1, y2; // opposite vertices of the rectangle


	if (feature->total == 0) {
		return NOT_DETECTED;
	}
	else {
		// find the rectangle with max area and assign it to that feature
		int x1_temp[feature->total];
		int y1_temp[feature->total];
		int x2_temp[feature->total];
		int y2_temp[feature->total];
		int area_temp[feature->total];
		index_max_area = 0;
		int max_area_temp = 0;
		for (int i = 0; i < (feature ? feature->total : 0); i++) {
			r = (CvRect*) cvGetSeqElem(feature, i);
			x1_temp[i] = r->x + featureROI.x;
			y1_temp[i] = r->y + featureROI.y;
			x2_temp[i] = x1_temp[i] + r->width;
			y2_temp[i] = y1_temp[i] + r->height;
			area_temp[i] = r->width * r->height;
			if (area_temp[i] > max_area_temp) {
				index_max_area = i;
				max_area_temp = area_temp[i];
			}
		}
		x1 = x1_temp[index_max_area];
		y1 = y1_temp[index_max_area];
		x2 = x2_temp[index_max_area];
		y2 = y2_temp[index_max_area];

		feature_box->x = x1;
		feature_box->y = y1;
		feature_box->width = abs(x2 - x1);
		feature_box->height = abs(y2 - y1);
		cvRectangle(displayImage, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(255, 0, 0), 1, 8, 0);
		return DETECTED;
	}
}


void FaceRegion::floodfill(CvRect floodfill_seed, CvScalar target_color, CvScalar replacement_color, CvRect bounding_box, vector<Point>* segment_vector) {
	//target color = hair color
 	//replacement color = blue

//	cvRectangleR(floodfillImage,floodfill_seed,CV_RGB(0,255,0)); // to display the green box used to scan in floodfill
//	cvNamedWindow("flood_fill_demo", 1);
//	cvMoveWindow("flood_fill_demo",100,50);
//	cvShowImage("flood_fill_demo", floodfillImage);
//	cvWaitKey(0);


	CvScalar seed_box_avg_color = getROIAverage(floodfillImage, floodfill_seed.x, floodfill_seed.y, floodfill_seed.width, floodfill_seed.height);
	double diff = euclideanColorDistance(seed_box_avg_color, target_color);
//	cout<<"difference: "<<diff<<endl;
	if (diff > 70) {
		//node color not equal to target color
//		cout<<"node color not equal to target color"<<endl;
	}
	else {
		//set node color to replacement color
		CvScalar current_pixel_color;
		for (int i = floodfill_seed.x; i < floodfill_seed.x + floodfill_seed.width; i++) {
			for (int j = floodfill_seed.y; j < floodfill_seed.y + floodfill_seed.height; j++) {
				current_pixel_color = cvScalar((double) pixel(floodfillImage, j, i, 0),	(double) pixel(floodfillImage, j, i, 1),	(double) pixel(floodfillImage, j, i, 2));
//				cout<<euclideanColorDistance(current_pixel_color, target_color)<<"\t";
				if (euclideanColorDistance(current_pixel_color, target_color) < 70) {
//					cout<<"i am here"<<endl;
					pixel(floodfillImage, j, i, 0) = replacement_color.val[0];
					pixel(floodfillImage, j, i, 1) = replacement_color.val[1];
					pixel(floodfillImage, j, i, 2) = replacement_color.val[2];

					// Add hair pixels to foreground for grabcut
					segment_vector->push_back(Point(i,j));
				}
			}
//			cout<<endl;
		}

		if (floodfill_seed.x >= bounding_box.x + floodfill_seed.width){
			CvRect west_box = cvRect(floodfill_seed.x, floodfill_seed.y, floodfill_seed.width, floodfill_seed.height);
			west_box.x = west_box.x - west_box.width;
			floodfill(west_box, target_color, replacement_color, bounding_box,segment_vector);
		}
		if (floodfill_seed.x <= bounding_box.x + bounding_box.width - 2 * floodfill_seed.width){
			CvRect east_box = floodfill_seed;
			east_box.x = east_box.x + east_box.width;
			floodfill(east_box, target_color, replacement_color, bounding_box,segment_vector);
		}
		if (floodfill_seed.y >= bounding_box.y + floodfill_seed.height) {
			CvRect north_box = floodfill_seed;
			north_box.y = north_box.y - north_box.height;
			floodfill(north_box, target_color, replacement_color, bounding_box,segment_vector);
		}
		if (floodfill_seed.y <= bounding_box.y + bounding_box.height - 2 * floodfill_seed.height) {
			CvRect south_box = floodfill_seed;
			south_box.y = south_box.y + south_box.height;
			floodfill(south_box, target_color, replacement_color, bounding_box,segment_vector);
		}
	}

//	const char* name = "floodfill";
//	cvNamedWindow(name, 1);
//	cvShowImage(name, img_color_hair_demo);
//	cvWaitKey(0);
//
//	cvReleaseImage(&img_color_hair_demo);
}

void FaceRegion::determineHairFloodfillBoundingBoxHeight(){
	int y_bottom_temp;
	if (flagLeftEyeDetect && flagRightEyeDetect) y_bottom_temp = MIN(eyeLeftRect.y, eyeRightRect.y);
	else if (!flagLeftEyeDetect && flagRightEyeDetect) y_bottom_temp = eyeRightRect.y;
	else if (flagLeftEyeDetect && !flagRightEyeDetect) y_bottom_temp = eyeLeftRect.y;

	hairFloodfillBoundingBox.height = (3*y_bottom_temp + faceRect.y)/4 - hairFloodfillBoundingBox.y;
}

