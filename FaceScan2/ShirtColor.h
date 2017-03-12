#pragma once
// ShirtDetection v1.2: Find the approximate color type of a person's tshirt. by Shervin Emami (shervin.emami@gmail.com), 30th Aug 2010.

// If trying to debug the color detector code, enable SHOW_DEBUG_IMAGE:
#define SHOW_DEBUG_IMAGE


#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
//#include <stdio.h>
//#include <tchar.h>

#include <cstdio>	// Used for "printf"
#include <string>	// Used for C++ strings
#include <iostream>	// Used for C++ cout print statements
//#include <cmath>	// Used to calculate square-root for statistics

// Include OpenCV libraries
#include "opencv2\highgui.hpp"
#include "opencv2\core.hpp"
#include "opencv2\objdetect.hpp"
//#include <cv.h>
//#include <cvaux.h>
//#include <cxcore.h>
//#include <highgui.h>

#include "ImageUtils.h"		// Used for easy image cropping, resizing, rotating, etc.

class ShirtColor
{
public:
	vector<CvRect> findObjectsInImage(IplImage *origImg, CvHaarClassifierCascade* cascade, CvSize minFeatureSize = cvSize(20, 20));
	int getPixelColorType(int H, int S, int V);
	void run();

private:
	ImageUtils _imageUtils;
};