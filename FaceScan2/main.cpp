#include "faceScan.h"
#include "ShirtColor.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "iostream"

using namespace cv;
using namespace std;

/* -------------------------- dlib file --------------------- */
string shape_landmark_dat_path = "assets/shape_predictor_68_face_landmarks.dat";

/* -------------------------- input file --------------------- */
string image_path = "assets/mild.png";

faceScan _faceScan;
ShirtColor _shirtColor;

int main(int nArgs, char* args[]) {
	/*
	// Image
	Mat image_front = imread(image_path);
	if (image_front.empty()) {
		cout << "Unable to open front image : " + image_path << "\n";
		system("pause");
		return 0;			
	}
	dlib::full_object_detection odt;
	std::vector<Point> points;

	_faceScan.detectFaceLandmark(image_front, odt);
	_faceScan.getPoints(odt, points);
	_faceScan.drawPoints(image_front, points);
	
	imshow("front", image_front);

	*/
	_shirtColor.run();
	Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
								  // open selected camera using selected API
	cap.open(deviceID + apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		// show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);
		
		if (waitKey(5) >= 0)
			break;
	}

	

	waitKey(0);

	return 0;
}