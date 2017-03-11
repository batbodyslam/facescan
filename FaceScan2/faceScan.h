#pragma once

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "iostream"

using namespace dlib;
using namespace cv;
using namespace std;

#define at_b(mat,r,c) mat.at<uchar>(r,c)
#define at_3b(mat,r,c) mat.at<Vec3b>(r,c)


class faceScan
{
public :
	 int findBiggestContour(std::vector<std::vector<Point> > contours);
	 void detectSkinColor(Mat &src, Mat &out_mask);
	 void drawPoints(Mat& inoutput, std::vector<cv::Point> &points);
	 void detectFaceLandmark(Mat &image, full_object_detection& out);
	 void getPoints(const full_object_detection &s, std::vector<cv::Point> &points);

	/* -------------------------- dlib file --------------------- */
	string shape_landmark_dat_path = "assets/shape_predictor_68_face_landmarks.dat";

	
};
