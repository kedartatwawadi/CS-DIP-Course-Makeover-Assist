/*! \class Accessories
    \brief adding accessories to the image
	
	\Description
    The function performs the task of adding glasses to the input image.<br>
    However the results are not very accurate if the eyes of the subject in the image<br>
    is not perfectly parallel to the horizontal or they dont lie in same plane. Only one<br>
    template of the glasses has been used. For enabling the glasses a trackbar has been <br>
    provided where '1' signifies selecting the glass and '0' means removing it.<br>

*/



#ifndef ACCESSORIES_H_
#define ACCESSORIES_H_

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utility_functions.h"
#include "FaceRegion.h"

using namespace std;
using namespace cv;

class Accessories {
public:

	//! Accessories  default constructor
		/*!
		A default constructor
		*/
	Accessories();

	//! Accessories constructor
	/*!
	   The constructor initialises data members and calls the put_glass function<br>
           to place glasses on the image and save it as modifiedImage.<br>
	   
	   \param face_region: a FaceRegion object, which contains info about the face features.<br>
	   Info like left and right eyebox would only be used.
	   */
	Accessories(FaceRegion* face_region, IplImage* glass_image);
	/// Contains the original image of the subject
	IplImage* image;
	/// Bounding box of the eye pair, not very useful always
	CvRect eyePair; 
	/// The left eye bounding rectangle
	CvRect eyeLeft;
	/// The right eye bounding rectangle
	CvRect eyeRight;
	/// The box bounding the face.
	CvRect faceBox;
	/// The glass image that will be used.
	IplImage* glass;
	/// The new image will be saved here.  
	IplImage* modifiedImage;
	/// To indicate if the left eye has been detected.
	int flagLeftEyeFound;
	/// To indicate if the right eye has been detected.
	int flagRightEyeFound;

	/// It intializes the data members of the class
	void initializeParams(FaceRegion* face_region, IplImage* glass_image);

	/// Performs the actual function of inserting the glass in the image.
	void put_glass(IplImage* original_img);
};


#endif /* ACCESSORIES_H_ */
