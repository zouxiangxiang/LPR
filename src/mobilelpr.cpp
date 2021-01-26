
#include "mobilelpr.hpp"
#include <time.h>
#include <opencv2/opencv.hpp>

LFFD::LFFD() {

}

LFFD::~LFFD() {

}

int LFFD::init(std::string model_path) {

	std::string param = model_path + "/det3-opt.param";
	std::string bin = model_path + "/det3-opt.bin";
	lffd.load_param(param.data());
	lffd.load_model(bin.data());

}

ncnn::Mat LFFD::plate_preprocess(ncnn::Mat img, PlateInfo &info) {
	cv::Mat gh(img.h, img.w, CV_8UC3);
	img.to_pixels(gh.data, ncnn::Mat::PIXEL_BGR);
	cv::Point2f srcTri[4];
	cv::Point2f dstTri[4];
	cv::Mat rot_mat(2, 4, CV_32FC1);
	cv::Mat warp_mat(2, 4, CV_32FC1);
	for (int j = 0; j < 4; j++) {
		srcTri[j] = cv::Point2f(info.landmark[2 * j], info.landmark[2 * j + 1]);
	}
    
	/*	int padding_x = cvFloor(h * 0.04 * 5);
	int padding_y = cvFloor(h * 0.04 * 2);*/
	int padding_x = 5;
	int padding_y = 3;
	int x0 = 0;		int y0 = 0;
	int x1 = 136;	int y1 = 0;
	int x2 = 136;	int y2 = 36;
	int x3 = 0;		int y3 = 36;

	dstTri[0] = cv::Point2f(x0 + padding_x, y0 + padding_y);
	dstTri[1] = cv::Point2f(x1 + padding_x, y1 + padding_y);
	dstTri[2] = cv::Point2f(x2 + padding_x, y2 + padding_y);
	dstTri[3] = cv::Point2f(x3 + padding_x, y3 + padding_y);

	warp_mat = cv::getAffineTransform(srcTri, dstTri);
	cv::Mat warp_dstImage = cv::Mat::zeros(36 + 2 * padding_y, 136 + 2 * padding_x, gh.type());
	cv::warpAffine(gh, warp_dstImage, warp_mat, warp_dstImage.size());
	ncnn::Mat sample = ncnn::Mat::from_pixels(warp_dstImage.data, ncnn::Mat::PIXEL_BGR, warp_dstImage.cols, warp_dstImage.rows);
	return sample;
}

void LFFD::crop_plates(ncnn::Mat input, std::vector<PlateInfo> &filters) {
	for (auto &info : filters) {
		info.license_plate = plate_preprocess(input, info);
	}
}



void LFFD::bbox_pad(std::vector<PlateInfo> &bboxes, int width, int height) {
	for (int i = 0; i < bboxes.size(); ++i) {
		ObjectBox &bbox = bboxes[i].bbox;
		bbox.xmin = round(std::max(bbox.xmin, 1.0f));
		bbox.ymin = round(std::max(bbox.ymin, 1.0f));
		bbox.xmax = round(std::min(bbox.xmax, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymax, height - 1.f));
	}
}


void LFFD::bbox_pad_square(std::vector<PlateInfo> &bboxes, int width, int height) {
	for (int i = 0; i < bboxes.size(); ++i) {

		ObjectBox &bbox = bboxes[i].bbox;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		float side = h > w ? h : w;
		float side_w = h > w ? h : w;
		float side_h = 12 / 30.0 * side_w; // 24,60, 48,120

		bbox.xmin = round(std::max(bbox.xmin + (w - side_w) * 0.5f, 0.f));
		bbox.ymin = round(std::max(bbox.ymin + (h - side_h) * 0.5f, 0.f));
		bbox.xmax = round(std::min(bbox.xmin + side_w - 1, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymin + side_h - 1, height - 1.f));
	}
}

bool CompareBBox(const PlateInfo &a, const PlateInfo &b) {
	return a.bbox.score > b.bbox.score;
}
std::vector<PlateInfo>LFFD::NMS(std::vector<PlateInfo> &bboxes, float thresh, char methodType) {
	std::vector<PlateInfo> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}
		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		ObjectBox select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) *
			(select_bbox.ymax - select_bbox.ymin + 1));
		float x1 = static_cast<float>(select_bbox.xmin);
		float y1 = static_cast<float>(select_bbox.ymin);
		float x2 = static_cast<float>(select_bbox.xmax);
		float y2 = static_cast<float>(select_bbox.ymax);

		select_idx++;
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			ObjectBox &bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}

std::vector<PlateInfo> LFFD::align(const ncnn::Mat &sample, std::vector<PlateInfo> &bbox) {
	std::vector<PlateInfo> aligns;
	int batch_size = bbox.size();

	for (int n = 0; n < batch_size; ++n) {
		ObjectBox &box = bbox[n].bbox;

		ncnn::Mat img_t, in;
		box.xmin = box.xmin < 0 ? 0 : box.xmin;
		box.xmax = box.xmax > sample.w ? sample.w : box.xmax;
		box.ymin = box.ymin < 0 ? 0 : box.ymin;
		box.ymax = box.ymax > sample.h ? sample.h : box.ymax;
		copy_cut_border(sample, img_t, box.ymin, sample.h - box.ymax, box.xmin, sample.w - box.xmax);
		resize_bilinear(img_t, in, width, height);
		ncnn::Extractor ex = lffd.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
        clock_t start, finish;
		
		ncnn::Mat score, bbox, point;
		start = clock();
		for(int kk=0; kk<10;kk++){
		
		ex.extract("prob1", score);
		ex.extract("conv6-2", bbox);
		ex.extract("conv6-3", point);}
        finish = clock();
	    std::cout << "The run time1 is: " << (double)(finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
		float conf = score[1];
		if (conf >= prob_threhold) {
			PlateInfo info;
			info.bbox.score = conf;
			info.bbox.xmin = box.xmin;
			info.bbox.ymin = box.ymin;
			info.bbox.xmax = box.xmax;
			info.bbox.ymax = box.ymax;

			for (int i = 0; i < 4; ++i) {
				info.bbox_reg[i] = bbox[i];
			}

			float w = info.bbox.xmax - info.bbox.xmin + 1.f;
			float h = info.bbox.ymax - info.bbox.ymin + 1.f;
			// x x x x y y y y to x y x y x y x y
			for (int i = 0; i < 4; ++i) {
				info.landmark[2 * i] = point[2 * i] * w*1.05 + info.bbox.xmin;
				info.landmark[2 * i + 1] = point[2 * i + 1] * h*1.1 + info.bbox.ymin;
			}
			aligns.push_back(info);
		}
	}

	return NMS(aligns, iou_threhold, 'm');
}

std::vector<PlateInfo> LFFD::align_plates(ncnn::Mat input, std::vector<PlateInfo> &filters) {
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	ncnn::Mat sample = input.clone();
	sample.substract_mean_normalize(mean_vals, norm_vals);
	std::vector<PlateInfo> plates = align(sample, filters);
	bbox_pad_square(plates, input.w, input.h);
	bbox_pad(plates, input.w, input.h);
	return  plates;
}

void LFFD::plate_detect(ncnn::Mat input, std::vector<PlateInfo_> face_info, std::vector<PlateInfo> &plates) {
	std::vector<PlateInfo> detect_plates;
	for (int i = 0; i < face_info.size(); i++) {
		std::cout << face_info.size() << std::endl;
		float rect_width = face_info[i].x2 - face_info[i].x1;
		float rect_height = face_info[i].y2 - face_info[i].y1;
		PlateInfo plateInfo;
		plateInfo.bbox.xmin = face_info[i].x1 - 0.2 * rect_width;
		plateInfo.bbox.ymin = face_info[i].y1 - 0.2* rect_height;
		plateInfo.bbox.xmax = face_info[i].x2 + 0.2 * rect_width;
		plateInfo.bbox.ymax = face_info[i].y2 + 0.2* rect_height;
		plateInfo.bbox.score = face_info[i].score;
		detect_plates.push_back(plateInfo);
	}
	plates = align_plates(input, detect_plates);
	crop_plates(input, plates);
}

