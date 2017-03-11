#pragma once

#include "dlib\opencv.h"
#include "dlib\image_processing\frontal_face_detector.h"
#include "dlib\image_processing\render_face_detections.h"
#include "dlib\image_processing.h""
#include "dlib\gui_widgets.h"

#include "opencv2\imgproc.hpp"
#include "opencv2\videoio.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\core.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "iostream"
#include <guiddef.h>

using namespace dlib;
using namespace cv;
using namespace std;

#define SHOW_PROCESS true
#define at_b(mat,r,c) mat.at<uchar>(r,c)
#define at_3b(mat,r,c) mat.at<Vec3b>(r,c)
#define NUM_MODELS 21
#define NUM_PARTS 68
#define NUM_FRONT_PARAMS  11
#define NUM_SIDE_PARAMS 2
#define ENABLE_ASSERTS

class faceScan
{
public :
	void setup();
	int findBiggestContour(std::vector<std::vector<Point> > contours);
	void detectSkinColor(Mat &src, Mat &out_mask);
	bool detectSideFace(Mat &img, std::vector<Rect_<int> > &side_faces);
	void drawPoints(Mat& inoutput, std::vector<cv::Point> &points);
	void detectFaceLandmark(Mat &image, full_object_detection& out);
	void getPoints(const full_object_detection &s, std::vector<cv::Point> &points);
	void calFrontFaceParams(Mat &image, std::vector<double> &prams, bool isShowProcess = false);
	void calSideFaceParams(Mat &image, std::vector<double> &prams, bool isShowProcess = false);
	void calDatabaseParams(std::vector<std::vector<double>>& base_front_params, std::vector<std::vector<double>>& base_side_params);
	void loadDatabaseParams(std::vector<std::vector<double>>& base_front_params, std::vector<std::vector<double>>& base_side_params);
	int findClosestModel(std::vector<std::vector<double>>& base_front_params, std::vector<std::vector<double>>& base_side_params
		, std::vector<double> &front_prams, std::vector<double> &side_prams);

	/* -------------------- CascadeClassifier -------------------- */
	CascadeClassifier frontal_face_cascade, side_face_cascade, eye_cascade;
	string frontal_face_cascade_path = "assets/haarcascades/haarcascade_frontalface_alt.xml";
	// string frontal_face_cascade_path = "assets/lbpcascades/lbpcascade_frontalface.xml";
	string side_face_cascade_path = "assets/haarcascades/haarcascade_profileface.xml";
	//string side_face_cascade_path = "assets/haarcascades/lbpcascade_profileface.xml";
	string eye_cascade_path = "assets/haarcascades/haarcascade_eye.xml";

	/* -------------------------- dlib file --------------------- */
	string shape_landmark_dat_path = "assets/shape_predictor_68_face_landmarks.dat";

	/* -------------------------- input file --------------------- */
	string base_prams_path = "assets/base/dataset/base_params.txt";
	string base_folder_path = "assets/base/dataset/";
	string front_image_path;
	string side_image_path;

	
};
