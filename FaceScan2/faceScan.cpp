#include "faceScan.h"


int faceScan::findBiggestContour(std::vector<std::vector<Point> > contours) {
	int indexOfBiggestContour = -1;
	int sizeOfBiggestContour = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size() > sizeOfBiggestContour) {
			sizeOfBiggestContour = contours[i].size();
			indexOfBiggestContour = i;
		}
	}
	return indexOfBiggestContour;
}

void faceScan::detectSkinColor(Mat &src, Mat &out_mask) {
	Mat frame(src.clone());
	Mat frame_gray;

	/* THRESHOLD ON HSV*/
	cvtColor(frame, frame, CV_BGR2HSV);
	GaussianBlur(frame, frame, Size(7, 7), 1, 1);
	medianBlur(frame, frame, 15);
	inRange(frame, Scalar(0, 48, 80), Scalar(20, 255, 255), frame_gray);

	morphologyEx(frame_gray, frame_gray, CV_MOP_ERODE, Mat1b(3, 3, 1), Point(-1, -1), 3);
	morphologyEx(frame_gray, frame_gray, CV_MOP_OPEN, Mat1b(7, 7, 1), Point(-1, -1), 1);
	morphologyEx(frame_gray, frame_gray, CV_MOP_CLOSE, Mat1b(9, 9, 1), Point(-1, -1), 1);

	medianBlur(frame_gray, frame_gray, 15);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	findContours(frame_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	int s = findBiggestContour(contours);

	out_mask = Mat::zeros(src.size(), CV_8UC1);
	cv::drawContours(out_mask, contours, s, Scalar(255), -1, 8, hierarchy, 0, Point());
}

void faceScan::drawPoints(Mat& inoutput, std::vector<cv::Point> &points) {
	for (int i = 0; i < points.size(); i++)
		cv::circle(inoutput, points[i], 3, Scalar(0, 255, 255), -1);
}

void faceScan::detectFaceLandmark(Mat &image, full_object_detection& out) {
	out = full_object_detection();
	std::vector<full_object_detection> shapes;
	try
	{
		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize(shape_landmark_dat_path) >> pose_model;

		// Grab a frame
		cv::Mat temp(image.clone());
		cv_image<bgr_pixel> cimg(temp);

		// Detect faces 
		std::vector<dlib::rectangle> faces = detector(cimg);
		// Find the pose of each face.
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));
		render_face_detections(shapes);
	}
	catch (serialization_error& e)
	{
		std::cout << "You need dlib's default face landmarking model file to run this example." << endl;
		std::cout << "You can get it from the following URL: " << endl;
		std::cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		std::cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		std::cout << e.what() << endl;
	}
	if (shapes.size()> 0)
		out = shapes[0];
}

void faceScan::getPoints(const full_object_detection &s, std::vector<cv::Point> &points) {
	points = std::vector<cv::Point>(s.num_parts());
	for (int i = 0; i < (int)s.num_parts(); i++) {
		dlib::point p = s.part(i);
		points[i] = cv::Point(p.x(), p.y());
	}
}


