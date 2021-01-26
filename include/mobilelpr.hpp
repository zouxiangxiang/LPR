#pragma once
#include "net.h"
#include "gpu.h"
//#include "base.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

//#include "plate_info.h"

#define NMS_UNION 1
#define NMS_MIN  2


typedef struct PlateInfo_ {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;

	float *landmarks;
} PlateInfo_;

typedef struct ObjectBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	float area;
} ObjectBox;

typedef struct PlateInfo {
	ObjectBox bbox;
	std::string plate_no;
	std::string plate_color;

	float bbox_reg[4];
	float landmark_reg[8];
	float landmark[8];

	ncnn::Mat license_plate;
	float confidence = 0;
} PlateInfo;


class LFFD {
public:

	LFFD();
	~LFFD();
	int init(std::string model_path);
	void plate_detect(ncnn::Mat input, std::vector<PlateInfo_> face_info, std::vector<PlateInfo> &plates);

private:

	ncnn::Mat plate_preprocess(ncnn::Mat img, PlateInfo &info);
	void crop_plates(ncnn::Mat input, std::vector<PlateInfo> &filters);
	void bbox_pad(std::vector<PlateInfo> &bboxes, int width, int height);
	void bbox_pad_square(std::vector<PlateInfo> &bboxes, int width, int height);
	std::vector<PlateInfo>NMS(std::vector<PlateInfo> &bboxes, float thresh, char methodType);
	std::vector<PlateInfo>align(const ncnn::Mat &sample, std::vector<PlateInfo> &bbox);
	std::vector<PlateInfo>align_plates(ncnn::Mat input, std::vector<PlateInfo> &filters);

private:
	ncnn::Net lffd;
	float iou_threhold=0.5;
	float prob_threhold=0.9;
	size_t width=120;
	size_t height=48;
	//int num_thread;
	//int num_output_scales;
	//int image_w;
	//int image_h;

	//std::string param;
	//std::string bin;

	//std::vector<float> receptive_field_list;
	//std::vector<float> receptive_field_stride;
	//std::vector<float> bbox_small_list;
	//std::vector<float> bbox_large_list;
	//std::vector<float> receptive_field_center_start;
	//std::vector<float> constant;

	//std::vector<std::string> output_blob_names;

	//const float mean_vals[3] = { 127.5, 127.5, 127.5 };
	//const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
};
