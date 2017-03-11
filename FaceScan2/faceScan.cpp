#include "faceScan.h"

/* ------------------ pair index for front faces ------------- */
int first_idx[] = { 0, 2, 4, 6, 37, 44, 8, 62, 31, 33, NUM_PARTS + 1 }; // size must be equals NUM_FRONT_PARAMS
int second_idx[] = { 16, 14, 12, 10, 41, 46, 66, 33, 35, NUM_PARTS, NUM_PARTS + 2 }; // size must be equals NUM_FRONT_PARAMS
double front_weight[] = { 1.0, 1.0, 1.2, 1.5, 3, 3, 3, 5, 5, 2, 2 };
double side_weight[] = { 0.0, 0.0 };


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

bool faceScan::detectSideFace(Mat &img, std::vector<Rect_<int> > &side_faces) {
	side_face_cascade.load(side_face_cascade_path);
	if (!side_face_cascade.empty()) {
		side_face_cascade.detectMultiScale(img, side_faces, 1.15, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		return side_faces.size() != 0;
	}
	std::cout << "Cannot Load Profileface Cascade File\n";
	return false;
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

void faceScan::calFrontFaceParams(Mat &image, std::vector<double> &prams, bool isShowProcess ) {
	full_object_detection shape;
	detectFaceLandmark(image, shape);
	if (shape.num_parts() == NUM_PARTS) {
		prams = std::vector<double>();
		//Mat front_shapes(image.size(), CV_8UC3);
		Mat image_front_shapes(image.clone());
		std::vector<cv::Point> points;
		getPoints(shape, points);
		// create top point
		cv::Point top = (points[19] + points[24]) / 2;
		points.push_back(top);

		// create left eye points
		cv::Point left_eye = (
			points[36] + points[37] + points[38] +
			points[39] + points[40] + points[41]
			) / 6;
		points.push_back(left_eye);
		// create right eye points
		cv::Point right_eye = (
			points[42] + points[43] + points[44] +
			points[47] + points[46] + points[45]
			) / 6;
		points.push_back(right_eye);
		for (int i = 0; i < NUM_FRONT_PARAMS; i++) {
			cv::Point a = points[first_idx[i]];
			cv::Point b = points[second_idx[i]];
			double dist = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
			prams.push_back(dist);
			line(image_front_shapes, a, b, Scalar(0, 0, 255));
		}
		double base_value = prams[0];
		for (int i = 0; i < NUM_FRONT_PARAMS; i++) {
			prams[i] = prams[i] / base_value;
			//cout << "front params : " << first_idx[i] << "," << second_idx[i] << "=" << prams[i] << endl;
		}
		//drawPoints(front_shapes, points);
		drawPoints(image_front_shapes, points);
		if (isShowProcess) {
			//cout << "points count :" << points.size();
			//imshow("front_shapes", front_shapes);
			imshow("image_front_shapes", image_front_shapes);
		}

	}

}

void faceScan::calSideFaceParams(Mat &image, std::vector<double> &prams, bool isShowProcess ) {
	Mat side_skin;
	detectSkinColor(image, side_skin);
	Mat front_line = Mat(side_skin.size(), CV_8UC3);
	front_line = Vec3b(0, 0, 0);
	int r_nose = 0, c_nose = side_skin.cols;
	std::vector<int> line_col(side_skin.rows);
	for (int r = 0; r < side_skin.rows; r++)
		for (int c = 0; c < side_skin.cols; c++)
			if (at_b(side_skin, r, c) == 255) {
				at_3b(front_line, r, c) = Vec3b(0, 255, 0);
				line_col[r] = c;
				if (c < c_nose)
				{
					r_nose = r;
					c_nose = c;
				}
				break;
			}
	for (int r = r_nose - 2; r >= 0; r--) {
		if (line_col[r] >= line_col[r + 1])
			at_3b(front_line, r, line_col[r]) = Vec3b(255, 0, 255);
		else break;
	}
	if (isShowProcess) {
		cv::circle(front_line, Point(c_nose, r_nose), 10, Scalar(255, 0, 0));
		imshow("side_skin", side_skin);
	}

	int low_tip = r_nose + side_skin.rows / 35;
	double nearly_threshold = side_skin.rows / 140;
	double left_most = side_skin.cols, right_most = 0;
	int r_chin = low_tip, r_neck = low_tip;
	for (int r = low_tip; r < side_skin.rows; r++) {
		if (line_col[r] == 0) continue;
		double end_of_line = line_col[r_nose] + 1.0*(side_skin.rows - r_nose) / (r - r_nose)*(line_col[r] - line_col[r_nose]);
		if (end_of_line < left_most || (r > r_chin && end_of_line < left_most + nearly_threshold)) {
			left_most = end_of_line;
			r_chin = r; // Chin Point
		}
		if (end_of_line > right_most + nearly_threshold*(r - r_neck) || r_chin > r_neck) {
			right_most = end_of_line;
			r_neck = r; // Neck Point
		}
	}
	if (isShowProcess) {
		line(front_line, Point(line_col[r_nose], r_nose), Point(left_most, side_skin.rows), Scalar(0, 0, 255));
		line(front_line, Point(line_col[r_nose], r_nose), Point(right_most, side_skin.rows), Scalar(255, 255, 0));
		circle(front_line, Point(line_col[r_chin], r_chin), 10, Scalar(0, 0, 255));
		circle(front_line, Point(line_col[r_neck], r_neck), 10, Scalar(255, 255, 0));
		imshow("front line", front_line);
	}

	double norm_fac = r_chin - r_nose;
	prams.push_back((double)(line_col[r_neck] - line_col[r_chin]) / norm_fac);
	prams.push_back((double)(line_col[r_neck] - line_col[r_nose]) / norm_fac);

}

void faceScan::calDatabaseParams(std::vector<std::vector<double>>& base_front_params, std::vector<std::vector<double>>& base_side_params) {
	base_front_params = std::vector<std::vector<double>>(NUM_MODELS);
	base_side_params = std::vector<std::vector<double>>(NUM_MODELS);
	for (int i = 0; i < NUM_MODELS; i++)
	{
		string front_path = base_folder_path + std::to_string(i) + "_front.jpg";
		string side_path = base_folder_path + std::to_string(i) + "_side.jpg";
		Mat front_img = imread(front_path);
		Mat side_img = imread(side_path);
		if (!front_img.empty())
			calFrontFaceParams(front_img, base_front_params[i]);
		if (!side_img.empty())
			calSideFaceParams(side_img, base_side_params[i]);
		std::cout << i + 1 << "/" << NUM_MODELS << " done\n";
	}
	std::cout << "Writing params to \"" << base_prams_path << "\" file\n";
	ofstream out_file(base_prams_path);
	if (out_file.is_open())
	{
		for (int i = 0; i < NUM_MODELS; i++) {
			out_file << base_front_params[i].size() << " ";
			for (int j = 0; j < base_front_params[i].size(); j++)
				out_file << fixed << base_front_params[i][j] << " ";
			out_file << "\n";
			out_file << base_side_params[i].size() << " ";
			for (int j = 0; j < base_side_params[i].size(); j++)
				out_file << fixed << base_side_params[i][j] << " ";
			out_file << "\n";
		}
		out_file.close();
	}
	else {
		std::cout << "Unable to open and write file !!!" << "\n";
	}
}

void faceScan::loadDatabaseParams(std::vector<std::vector<double>>& base_front_params, std::vector<std::vector<double>>& base_side_params) {
	base_front_params = std::vector<std::vector<double>>(NUM_MODELS);
	base_side_params = std::vector<std::vector<double>>(NUM_MODELS);
	std::cout << "Reading params from \"" << base_prams_path << "\" file\n";
	ifstream in_file(base_prams_path);
	if (in_file.is_open())
	{
		for (int i = 0; i < NUM_MODELS; i++) {
			int front_params_size, side_params_size;
			in_file >> front_params_size;
			base_front_params[i].assign(front_params_size, 0);
			for (int j = 0; j < front_params_size; j++) {
				in_file >> base_front_params[i][j];
				std::cout << base_front_params[i][j] << " ";
			}
			std::cout << "\n";
			in_file >> side_params_size;
			base_side_params[i].assign(side_params_size, 0);
			for (int j = 0; j < side_params_size; j++) {
				in_file >> base_side_params[i][j];
				std::cout << base_side_params[i][j] << " ";
			}
			std::cout << "\n";
			std::cout << i + 1 << "/" << NUM_MODELS << " done\n";
		}
		in_file.close();
	}
	else {
		std::cout << "Unable to open and read file !!!" << "\n";
	}
}

int faceScan::findClosestModel(std::vector<std::vector<double>>& base_front_params, std::vector<std::vector<double>>& base_side_params
	, std::vector<double> &front_prams, std::vector<double> &side_prams)
{
	/// TODO: PLEASE IMPLEMENT IT.
	std::vector<pair<double, int>> mse(NUM_MODELS, make_pair(-1, -1));

	for (int i = 0; i < NUM_MODELS; i++) {
		mse[i].second = i;
		if (base_front_params[i].size() != NUM_FRONT_PARAMS || base_side_params[i].size() != NUM_SIDE_PARAMS)
			continue;

		double sum_e_sq = 0;
		for (int j = 0; j < NUM_FRONT_PARAMS; j++) {
			double e = (base_front_params[i][j] - front_prams[j]);
			double e_sq = e*e;
			sum_e_sq += e_sq * front_weight[j];
		}
		for (int j = 0; j < NUM_SIDE_PARAMS; j++) {
			double e = (base_side_params[i][j] - side_prams[j]);
			double e_sq = e*e;
			sum_e_sq += e_sq * side_weight[j];;
		}

		mse[i].first = sum_e_sq / (NUM_FRONT_PARAMS + NUM_SIDE_PARAMS);

	}
	std::sort(mse.begin(), mse.end());

	std::cout << "Sorted MSE : \n";
	for (int i = 0; i < NUM_MODELS; i++)
	{
		std::cout << "rank: " << (i + 1) << ":: " << mse[i].second << " -> " << mse[i].first << "\n";
	}
	std::cout << "\n";
	return mse[0].second;
}
