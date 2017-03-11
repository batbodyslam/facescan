//#define ENABLE_ASSERTS

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

#include "faceScan.h"

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

faceScan _faceScan;

int main(int nArgs, char* args[]) {

	if (nArgs == 2 && !strcmp(args[1], "recal")) {
		/* calulate base model params */
		std::vector<std::vector<double>> base_front_params, base_side_params;
		cout << "Starting..." << "\n";
		cout << "Calculating base models..." << "\n";
		_faceScan.calDatabaseParams(base_front_params, base_side_params);
		cout << "Completed !!!" << "\n";
		system("pause");
	}
	else {
		if (nArgs == 3) {
			front_image_path = args[1];
			side_image_path = args[2];
		}
		else {
			std::cout << "Please enter FRONT facial image path: ";
			std::cin >> front_image_path;
			std::cout << "Please enter SIDE facial image path: ";
			std::cin >> side_image_path;
		}

		/* load base model params */
		std::vector<std::vector<double>> base_front_params, base_side_params;
		std::cout << "Starting..." << "\n";
		std::cout << "Loading base models..." << "\n";
		_faceScan.loadDatabaseParams(base_front_params, base_side_params);
		std::cout << "Completed !!!" << "\n";

		/* Front*/
		Mat image_front = imread(front_image_path);
		if (image_front.empty()) {
			cout << "Unable to open front image : " + front_image_path << "\n";
			system("pause");
			return 0;
		}

		imshow("front", image_front);
		// Calulate front prams
		std::vector<double> front_prams;
		_faceScan.calFrontFaceParams(image_front, front_prams, SHOW_PROCESS);

		/*Left Side*/
		Mat image_side = imread(side_image_path);
		if (image_side.empty()) {
			cout << "Unable to open side image : " + side_image_path << "\n";
			system("pause");
			return 0;
		}
		Mat prepared_image_side;
		std::vector<Rect_<int> > side_faces;
		if (false && _faceScan.detectSideFace(image_side, side_faces)) {
			Rect side_face = side_faces[0];
			//side_face = Rect(side_face.x, std::max(side_face.y - side_face.height*0.1, 0.0),
			//	std::min((int)(side_face.width*1.3), image_side.cols), std::min((int)(side_face.width*1.3), image_side.rows));
			Mat side_roi = image_side(side_face);
			prepared_image_side = side_roi;
		}
		else {
			prepared_image_side = image_side;
		}
		imshow("side", prepared_image_side);
		// Calulate side prams
		std::vector<double> side_prams;
		_faceScan.calSideFaceParams(prepared_image_side, side_prams, SHOW_PROCESS);

		/* Find closest model*/
		int idx = _faceScan.findClosestModel(base_front_params, base_side_params, front_prams, side_prams);

		std::cout << "\n";
		std::cout << "#############################\n";
		std::cout << " The Closest Model is " + to_string(idx) << "\n";
		std::cout << "#############################\n";
		cout << (base_folder_path + std::to_string(idx) + "_front.jpg").c_str();
		Mat model_img = imread(base_folder_path + std::to_string(idx) + "_front.jpg");
		imshow("Model : " + std::to_string(idx), model_img);

		waitKey(0);

	}
	return 0;
}