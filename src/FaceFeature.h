#ifndef FACEFEATURE_H_
#define FACEFEATURE_H_

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

class FaceFeature {
public:
	FaceFeature(const char* feature_filename,double scale_factor,int min_neighbours,int flags,CvSize min_size);
	const char *featureHaarCascadeFilename;
	CvHaarClassifierCascade* featureCascade;
	double haarScaleFactor;
	int haarMinNeighbours;
	int haarFlags;
	CvSize minFeatureSize;
};

#endif /* FACEFEATURE_H_ */
