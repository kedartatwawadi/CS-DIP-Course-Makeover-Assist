/*! \class HairRegion
    \brief represents the properties of the hair region

The class represents the properties of hair region such as hue and saturation values
which are used for coloring the hair to a desired color<br>

*/
#ifndef HAIRREGION_H_
#define HAIRREGION_H_

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

class HairRegion {
public:
	/// Default constructor
	HairRegion();

	/// Constructor used when the image and the hair mask are given
	HairRegion(IplImage* original_image,IplImage* mask_image);

	/// holds the gray-scale image of the hair region as obtained form grabcut
	IplImage* mask;

	/// holds the original image of which the hair needs to be colored
	IplImage* originalImage;

	/// holds the output image after coloring the hair to its desired value
	IplImage* modifiedImage;

	/// represents the original hair color of the person in the image
	CvScalar originalHairColor;

	/// represents the hue to which the hair color should be changed to
	int requiredHairHue;

	/// represents the saturation to which the hair color should be changed to
	int requiredHairSaturation;

	//! sets the input and the mask images
	/*!
	\param original_image the input image on which hair coloring is to be performed
	\param mask_image the gray-scale image representing the hair region as
					obtained form the grabcut
	*/
	void setImageAndMask(IplImage* original_image,IplImage* mask_image);

	//! scales up the gray-scale mask
	/*!
	 *  scales up the gray-scale mask as obtained from grabcut and then performs
	 *  morphological operations, such as erosion and dilation, on it
	*/
	void scaleMask(); // scale the bin map of 0/3 to 0/255

	//! sets the hue and saturation values for the hair
	/*!
	\param hue the desired hue value of the hair
	\param saturation the desired saturation value of the hair
	*/
	void setRequiredHairHueSaturation(unsigned char hue, unsigned char saturation);

	//! changes the color of the hair
	/*!
	 * changes the color of the hair as per the hue and saturation values of the hair
	 * as given by the user
	*/
	void changeHueSaturation();
};

#endif /* HAIRREGION_H_ */
