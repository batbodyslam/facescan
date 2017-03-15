#pragma once
// ShirtDetection v1.2: Find the approximate color type of a person's tshirt. by Shervin Emami (shervin.emami@gmail.com), 30th Aug 2010.

// If trying to debug the color detector code, enable SHOW_DEBUG_IMAGE:
#define SHOW_DEBUG_IMAGE


#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
//#include <stdio.h>
//#include <tchar.h>

#include <cstdio>	// Used for "printf"
//#include <string>	// Used for C++ strings
#include <iostream>	// Used for C++ cout print statements
#include <cmath>	// Used to calculate square-root for statistics

// Include OpenCV libraries
#include "opencv2\highgui.hpp"
#include "opencv2\core.hpp"
#include "opencv2\objdetect.hpp"
#include "opencv\cv.hpp"
#include "opencv\cvaux.hpp"
#include "opencv\cxcore.hpp"
//#include <cv.h>
//#include <cvaux.h>
//#include <cxcore.h>
//#include <highgui.h>

#include "ImageUtils.h"		// Used for easy image cropping, resizing, rotating, etc.
using namespace std;
using namespace cv;

class ShirtColor
{
public:
	std::vector<Rect> findObjectsInImage(Mat origImg, CascadeClassifier cascade, Size minFeatureSize = Size(20, 20));
	int getPixelColorType(int H, int S, int V);
	void run();
	void readParams();

private:
	ImageUtils _imageUtils;
	// Face Detection HaarCascade Classifier file for OpenCV (downloadable from "http://alereimondo.no-ip.org/OpenCV/34").
	//const char* cascadeFileFace = "haarcascades\\haarcascade_frontalface_alt.xml";	// Path to the Face Detection HaarCascade XML file
	//const char* cascadeFileFace = "assets\\haarcascades\\haarcascade_frontalface_alt.xml";	// Path to the Face Detection HaarCascade XML file
	string cascade_path = "assets/haarcascades/haarcascade_frontalface_alt.xml";
	string image_path = "assets/test1.png";
	string params_path = "assets/params.xml";
	float SHIRT_DY ;	// Distance from top of face to top of shirt region, based on detected face height.
	float SHIRT_SCALE_X;	// Width of shirt region compared to the detected face
	float SHIRT_SCALE_Y;	// Height of shirt region compared to the detected face

};