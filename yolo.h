#pragma once
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "qtTest.h"


class qtTest;
class Output {
public:
	int id;//������id
	float confidence;//������Ŷ�
	cv::Rect box;//���ο�
};


class Yolo {
public:
	//��yolo.h�е� Yolo������ӳ�Ա����readModel��
	bool readModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
	void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);
	void disPlay();
private:
	//�����һ������
	float Sigmoid(float x) {
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}
	//anchors
	const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
	//stride
	const float netStride[3] = { 8.0, 16.0, 32.0 };
	const int netWidth = 640; //����ģ�������С
	const int netHeight = 640;
	float nmsThreshold = 0.45;
	float boxThreshold = 0.35;
	float classThreshold = 0.35;
	//����
	/*std::vector<std::string> className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };*/
	std::vector<std::string> className = {"house","ship","storage tank","baseball diamond","tennis court","basketball court",
	"ground track field","harbor","bridge","vehicle"};
	//std::vector<std::string> className = { "house","ship","storage tank"};
};
