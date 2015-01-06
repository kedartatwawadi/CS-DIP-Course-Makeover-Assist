#ifndef GRABCUT_H_
#define GRABCUT_H_

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FaceRegion.h"

using namespace std;
using namespace cv;

class GrabCut {
public:
	enum {NOT_SET = 0, IN_PROCESS = 1, SET = 2};
	enum {BACKGROUND = 0, FOREGROUND = 1};
	static const int radius = 2;
	static const int thickness = -1;
	void reset();
	void setRect(Rect bounding_box);
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage() const;
	void imageSegment(FaceRegion* face_region);
	Rect hair1;
	uchar hair1State;

private:
	void setRectInMask();
	void setPixelsInMask(int pixel_nature, Point p);
	void setRectOfPixels(Rect c, int pixel_nature);

	const string* winName;
	const Mat* image;
	Mat mask;
	Mat bgdModel, fgdModel;
	Rect rect;

	bool isInitialized;

	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
};

void getBinMask(const Mat& comMask, Mat& binMask);

#endif /* GRABCUT_H_ */
