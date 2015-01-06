#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "stdio.h"
#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FaceFeature.h"
#include "FaceRegion.h"
#include "GrabCut.h"
#include "HairRegion.h"
#include "utility_functions.h"
#include "Accessories.h"

using namespace std;
using namespace cv;

HairRegion hairRegion1;
Accessories accessories1;
int glass_enable;

void updateUIGlasses(int pos) {

	if (pos==1){
//		glass_enable = 1;
		accessories1.put_glass(hairRegion1.modifiedImage);
		cvShowImage("Adjust Hair Color", accessories1.modifiedImage);
	}
	else {
//		glass_enable = 0;
		cvShowImage("Adjust Hair Color", hairRegion1.modifiedImage);
	}
//	cout<<"glasses added: "<<pos<<endl;
//	return;
}

void updateUIHue(int pos) {
	hairRegion1.requiredHairHue = pos;
	hairRegion1.changeHueSaturation();
//	cvShowImage("Adjust Hair Color", hairRegion1.modifiedImage);
	updateUIGlasses(glass_enable);
}

void updateUISaturation(int pos) {
	hairRegion1.requiredHairSaturation = pos;
	hairRegion1.changeHueSaturation();
//	cvShowImage("Adjust Hair Color", hairRegion1.modifiedImage);
	updateUIGlasses(glass_enable);
}

int main(int argc, char** argv) {
	GrabCut grabcut1;

	IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
	IplImage* glass = cvLoadImage(argv[2], CV_LOAD_IMAGE_COLOR);

	string image_basename = getImageID(argv[1]);
	cout << "Checking for image : " << image_basename << endl;


	FaceRegion face_region1(img);
	face_region1.findAllFaceFeatures();
	cvNamedWindow("input image",1);
	cvMoveWindow("input image",100,100);
	cvShowImage("input image", face_region1.displayImage);
	while (cvWaitKey(10) > 0);

	accessories1.initializeParams(&face_region1, glass);

	Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (image.empty()) cout << "\n Durn, couldn't read image filename " << endl;
	const string winName = "grabcut";
	cvNamedWindow(winName.c_str(), 1);
	cvMoveWindow(winName.c_str(),600,100);
	grabcut1.setImageAndWinName(image, winName);
	grabcut1.imageSegment(&face_region1);
	grabcut1.showImage();
	while (cvWaitKey(10)>0);

	IplImage* mask1 = cvLoadImage("Data/mask.png", CV_LOAD_IMAGE_GRAYSCALE);
	hairRegion1.setImageAndMask(img, mask1);
	hairRegion1.scaleMask();

	cvNamedWindow("Adjust Hair Color", 1);
	cvMoveWindow("Adjust Hair Color",800,100);
	cvCreateTrackbar("Hue","Adjust Hair Color",&hairRegion1.requiredHairHue,180,updateUIHue);
	cvCreateTrackbar("Saturation","Adjust Hair Color",&hairRegion1.requiredHairSaturation,180,updateUISaturation);
	if ((accessories1.flagLeftEyeFound == 1)|| (accessories1.flagRightEyeFound == 1))
		cvCreateTrackbar("Glasses","Adjust Hair Color", &glass_enable, 1, updateUIGlasses);

	cvSetTrackbarPos("Hue","Adjust Hair Color",20);
	cvSetTrackbarPos("Saturation","Adjust Hair Color",120);

	cvWaitKey(0);

	cvNamedWindow("output",1);
	cvMoveWindow("output",200,200);
//	cvShowImage("output",hairRegion1.modifiedImage);
	if (glass_enable)
		cvShowImage("output", accessories1.modifiedImage);
	else
		cvShowImage("output", hairRegion1.modifiedImage);
	cvWaitKey(0);

	cvReleaseImage(&img);
	cvReleaseImage(&mask1);

	return 0;
}









/*!

 This is a makeover assist software project done by:<br>
 <b>
 Ayesha Mudassir [09007014]<br>
 Kedar Tatwawadi [09D07022]<br>
 Siddharth Sarangdhar[09026007]<br>
 Sudipto Mondal[09D07023]<br>
 </b>
 Under the guidance of <b>Prof Sharat Chandran</b> , CSE IIT Bombay <br>
 The project was undertaken as a part of <b> CS663: Fundamentals of Digital Image Processing</b> course.<br>
 \mainpage Makeover Assist Software
  \section intro_sec Introduction
  Our project idea is to create a software which would help in predicting how would the user look on
applying a makeover. <br>
We mainly concentrating on changing the hair color of the user.<br>
The other makeovers include, adding glasses.


  The flow diagram for the project is shown below:
  <ul>
  	  <li> Detect Frontal Face </li>
  	  <li> Detect Various Facial features</li>
  	  <li> Find a hair pixel seed for implementing flood fill</li>
  	  <li> Use modified flood fill algorithm to find representative hair pixels</li>
  	  <li> Tag the representative hair pixels as foreground</li>
  	  <li> Tag the eye, cheek, neck regions as background</li>
  	  <li> use grabcut segmentation algorithm to segment the hair region </li>
  	  <li> set the sliders to change the colors of the hair region </li>
  	  <li> set slider to insert glasses onto the image</li>
  </ul>

  <p>

  The following classes were implemented:
  <ul>
  	<li> FaceFeature      </li>
  	<li> FaceRegion     </li>
  	<li> HairRegion  </li>
  	<li> GrabCut     </li>
  	<li> Accessories  </li>
  </ul>
  Note that the class GrabCut is a modified class of GCApplication class from the OpenCV sample Grabcut application.



  \section sec_2 User Interface
  The user interface is as follows:<br>

  <br>
  We have implemented sliders for the following tasks<br>
  <ul>
  	<li>1 : changing Hue of HairRegion </li>
  	<li>2 : changing Saturation of HairRegion</li>
  	<li>3 : inserting glasses</li>
  </ul>




    </pre><br>

    The sample execution proceeds as follows:<br>
    <pre>

    ./sample_run.sh

    The sample_run.sh script is a simple script which does the following.
    --- [Cleans the project]
    --- [ compiles the program using the Makefile]
    --- [ Executes the executable on the images in the Data/sample directory]

    </pre><br>



	<p>
	 We explain the working of Hair Detection , Coloring algorithm using an example.


     We first detect various face features using Haar Like Classifier Cascades.<br>
     The features which we detect are:
     <ul>
     <li>Frontal Face</li>
     <li>Left Eye Region</li>
     <li>Right Eye Region</li>
     <li>Left Cheek>/li>
     <li>Right Cheek</li>
     <li> Forehead </li>
     </ul>
     After detecting the hair seed, we apply a modified flood fill algorithm to
     find more representative hair pixels.<br>

     <div align="center">
     <br><h3> Face Feature Detection and FloodFill on hair region.</h3>
     <img src="images/5.png" alt="Makeover Assist" width="600" align="middle"  />
     </div>

    After applying flood fill algorithm to find representative hair pixels,
    We apply GrabCut Algorithm to segment the HairRegion.<br>
    Our Grabcut Algorithm does the following:
    <ul>
    <li> Placing a bounding box around the face , specifying foreground</li>
    <li> Tagging pixels obtained from floodfill as foreground</li>
    <li> Tagging eye, cheek mouth, neck regions as background</li>
    <li> Applying 10 iterations of the Grabcut algorithm </li>
    </ul>
    The resultant mask obtained is shown in the 3 rd figure[rightmost].

    <div align="center">
    <br><h3> GrabCut Implementation for finding Hair Mask</h3>
    <img src="images/6.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>



    Once the mair mask is isolated, we can then apply Hair coloring.<br>
    Simple hair coloring methods can be changing the Hue of the hair region.<br>
    In the sample images given below, using the slider interface provided to change hue-saturation,
    we can color the hair to various colors<br>
    Example:<br>
    <ul>
    <li> Yellow</li>
    <li> Green</li>
    <li> Blue</li>
    <li> Purple</li>
    <li> Brown</li>
 	<li> Gray </li>
 	</ul>
 	etc.

    <div align="center">
    <br><h3> Changing Hue and Saturation of HairRegion</h3>
    <img src="images/7.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>

    <div align="center">
    <br><h3> Green Hair</h3>
    <img src="images/8.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>

    <div align="center">
    <br><h3> Blue Hair</h3>
    <img src="images/9.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>

    <div align="center">
    <br><h3> Purple Hair</h3>
    <img src="images/10.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>

    <div align="center">
    <br><h3> Brown Hair</h3>
    <img src="images/11.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>


    <div align="center">
    <br><h3> Gray Hair</h3>
    <img src="images/12.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>


    <div align="center">
    <br><h3> With Glasses</h3>
    <img src="images/14.png" alt="Makeover Assist" width="600" align="middle"  />
    </div>


     <br>
     </p>

Have a look at a demo video which shows the working of our demo program:




  \section  sec_further_imp Further Improvements
  The following improvements can be made to the code.
  <ol>
  	  <li> Better implementation of modified flood fill algorithm to detect more representative hair pixels <br>
  	  	   Would be useful to use a better color model for the hair region.</li>
  	  <li> Improvements can be made to Grabcut algorithm, by considering probalistic tagging of pixels</li>
  	  <li> Improvements to hair coloring. As its seen , the hair coloring doesnt look as natural for dark hair.
  	  	   This problem needs to be handled, by appropriately varying the intensity too.</li>
  	  <li> Another major improvement can be hair highlighting. Hair highlighting is a complex task in which we
  	   detect the hair texture, and then color a few strands of hair.
  	  </li>
  	  <li> Better GUI, more variety of accessories, can be a simple extension.
  	  </li>
  </ol>

  \section refer References
  <ol>
  	 <li> <b>OpenCV </b> http://opencv.willowgarage.com/documentation/cpp/index.html </li>
     <li> <b>Detection, Analysis and Matching of Hair] </b>Yaser Yacoob et al: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=01541327 </li>
     <li> <b>Grabcut Algorithm</b>  </li>
     <li> <b>GraphCut Algorithm</b> </li>
     <li> <b>FloodFill:</b> http://en.wikipedia.org/wiki/Flood_fill </li>
  </ol>

*/
