/*! \class FaceRegion
    \brief represents various facial features in the face region.

The class represents the facial features, and other properties which would be
further useful for grabcut algorithm,hair coloring and other applications<br>

*/


#ifndef FACEREGION_H_
#define FACEREGION_H_

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FaceFeature.h"

using namespace std;
using namespace cv;

class FaceRegion{
public:
	enum {NOT_DETECTED = 0, /// useful to set a particular Feature has been set or not.
		DETECTED = 1 /// useful for setting up flags.
	};


	/// holds the image of which the face and its features have to be found
	IplImage* image;

	/// the image to be displayed,shows rectangles around detected features.
	IplImage* displayImage;

	/// the image obtained after coloring the flood filled pixels as blue
	IplImage* floodfillImage;

	///the bounding box of the face
	Rect faceRect;

	/// the bounding box of the face, hair, etc. together
	Rect boundingRect;

	/// the bounding box of the left eye
	Rect eyeLeftRect;

	/// the bounding box of the right eye.
	Rect eyeRightRect;

	/// the bounding box of the mouth
	Rect mouthRect;

	/// the bounding box of the nose
	Rect noseRect;

	/// the bounding box for the eye pair.
	Rect eyePairRect;

	 //forehead
	Rect skinForehead;

	//left cheek
	Rect skinLeftCheek;

	//right cheek
	Rect skinRightCheek;


	vector<Point> Hair_vector;

	//flags

	/// set to DETECTED if face detected
	int flagFaceDetect;

	/// set to DETECTED if left eye detected.
	int flagLeftEyeDetect;

	/// set to DETECTED if Right eye detected.
	int flagRightEyeDetect;

	/// set to DETECTED if nose detected.
	int flagNoseDetect;

	/// set to DETECTED if mouth detected.
	int flagMouthDetect;

	/// set to DETECTED if the hair seed for flood fill is set.
	int flagHairSeedDetect;

	//averaging properties:
	/// the average forehead skin color.
	CvScalar avgForeheadSkinColor;

	/// the
	CvScalar avgLeftCheekSkinColor;


	CvScalar avgRightCheekSkinColor;


	CvScalar avgHairColor;


	CvScalar avgBackgroundColor;

	//implementation variables
	CvRect hairFloodfillSeed;


	CvRect hairFloodfillBoundingBox;


	CvRect backgroundFloodfillSeed;


	CvRect backgroundFloodfillBoundingBox;



	Rect leftBackgroundBox;


	Rect rightBackgroundBox;


	//constructor
	FaceRegion(IplImage* img);

	//functions
	void findAllFaceFeatures();


	void findFace();


	void findLeftEye();


	void findRightEye();


	void findNose();


	void findMouth();


	void findForehead();


	void findLeftCheek();


	void findRightCheek();


	void findAvgSkinColors();


	void findGrabcutBoundingRect();


	void findAvgHairColor();


	void tagHairPixels();


	void findLeftBackgroundBox();


	void findRightBackgroundBox();



	//functions



	//! detects a particular face feature.
	/*!
	\param featureROI the ROI region for the particular feature.
		   it helps to finetune the region where the feature can be found
	\param face_feature a FaceFeature object which it wants to detect.
		    leftEye, frontalFace etc.
	\param feature_box a Rect* pointer to the bounding box of the feature , which is set when the feature is found.
	\return int returns DETECTED if the feature was detected, else NOT_DETECTED
	\sa FaceFeature
	*/
	int detectFeature(CvRect featureROI, FaceFeature face_feature, Rect* feature_box);


	//! modified floodfill algo, finds a connected region of similar color.
		/*!

		 The modified floodfill

		\param floodfill_seed the .
			   it helps to finetune the region where the feature can be found
		\param face_feature a FaceFeature object which it wants to detect.
			    leftEye, frontalFace etc.
		\param feature_box a Rect* pointer to the bounding box of the feature , which is set when the feature is found.
		\return int returns DETECTED if the feature was detected, else NOT_DETECTED
		\sa FaceFeature

		*/
	void floodfill(CvRect floodfill_seed, CvScalar target_color, CvScalar replacement_color, CvRect bounding_box, vector<Point>* segment_vector);


	//! determines the bounding box necessary for flood fill algo.
	void determineHairFloodfillBoundingBoxHeight();
};

#endif /* FACEREGION_H_ */
