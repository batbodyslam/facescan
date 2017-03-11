#include "faceScan.h"

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

int main(int nArgs, char* args[]) {

	/* Image*/
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


	waitKey(0);

	return 0;
}