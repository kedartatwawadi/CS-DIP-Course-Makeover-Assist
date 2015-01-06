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

FaceFeature::FaceFeature(const char* feature_filename, double scale_factor, int min_neighbours, int flags, CvSize min_size) {
	featureHaarCascadeFilename = feature_filename;
	featureCascade = (CvHaarClassifierCascade*) cvLoad(featureHaarCascadeFilename, 0, 0, 0);
	haarScaleFactor = scale_factor;
	haarMinNeighbours = min_neighbours;
	haarFlags = flags;
	minFeatureSize = min_size;
}
