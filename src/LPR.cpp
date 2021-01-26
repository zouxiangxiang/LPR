//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "mobilelpr.hpp"
#include "Pipeline.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <time.h>
#include <cmath>
int main(int argc, char **argv) {
    // if (argc <= 3) {
    //     fprintf(stderr, "Usage: %s <ncnn bin> <ncnn param> [image files...]\n", argv[0]);
    //     return 1;
    // }

   prc::PipelinePR prc_("../data/LPRmodel/cascade.xml",
		"../data/LPRmodel/HorizonalFinemapping.prototxt", "../data/LPRmodel/HorizonalFinemapping.caffemodel",
		"../data/LPRmodel/Segmentation.prototxt", "../data/LPRmodel/Segmentation.caffemodel",
		"../data/LPRmodel/CharacterRecognization.prototxt", "../data/LPRmodel/CharacterRecognization.caffemodel",
		"../data/LPRmodel/SegmenationFree-Inception.prototxt", "../data/LPRmodel/SegmenationFree-Inception.caffemodel"
	);
    std::string bin_path = "../data/version-slim/ncnn_model_plr.bin";
    std::string param_path = "../data/version-slim/ncnn_model_plr.param";
    UltraFace ultraface(bin_path, param_path, 320, 240, 1, 0.7); // config model input
    std::string model_path ="../data/mobilelpr/models/float";
    LFFD lffd;
    lffd.init(model_path);
    
    for (int i = 3; i < 100; i++) {
        std::string image_file = "../data/imgae/1.jpg";
        std::cout << "Processing " << image_file << std::endl;
        
        cv::Mat frame = cv::imread(image_file);
        int width=static_cast<float>(frame.cols);
        int height=static_cast<float>(frame.rows);
        double scale=0.5;
        cv::resize(frame,frame,cv::Size(width*scale,height*scale));
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        ncnn::Mat sample = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
        clock_t start, finish;
        std::vector<FaceInfo> face_info;
        std::vector<PlateInfo> plates;
        std::vector<PlateInfo_> face_info_;
	    PlateInfo_ lk;
        start= clock();
        ultraface.detect(inmat, face_info);
        finish = clock();
	    std::cout << "The run time is: " << (double)(finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }

        //cv::imshow("UltraFace", frame);
       // cv::waitKey();
       // cv::imwrite("result.jpg", frame);
      for (int i = 0; i < face_info.size(); i++) {
		lk.x1 = face_info[i].x1;
		lk.x2 = face_info[i].x2;
		lk.y1= face_info[i].y1;
		lk.y2 = face_info[i].y2;
		lk.landmarks= face_info[i].landmarks;
		lk.score= face_info[i].score;
		face_info_.push_back(lk);
		auto face = face_info[i];
		cv::Rect region = cv::Rect(face.x1 - (face.x2 - face.x1)*0.1, face.y1 - (face.x2 - face.x1)*0.1, (face.x2 - face.x1)*1.3, (face.y2 - face.y1)*1.2);
		cv::Mat image_cut = frame(region);
		std::cout << "---------iiii--------:" << face_info[i].score << std::endl;
		clock_t t12 = clock();
	}

    start = clock();
	lffd.plate_detect(sample, face_info_, plates);
    finish = clock();
    std::cout << "The run time is: " << (double)(finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
    for (auto plateInfo : plates) {
		ncnn::Mat in = plateInfo.license_plate.clone();
		cv::Mat gh(in.h, in.w, CV_8UC3);
		in.to_pixels(gh.data, ncnn::Mat::PIXEL_BGR);
        start = clock();
		std::vector<prc::PlateInfo_> res = prc_.RunPiplineAsImage(gh, prc::SEGMENTATION_FREE_METHOD);
		finish = clock();
		std::cout << "The run time is: " << (double)(finish - start) / CLOCKS_PER_SEC << "s" << std::endl;
		for (auto st : res) {
			if (st.confidence > 0.75) {
				std::cout << st.getPlateName() << " " << st.confidence << std::endl;
			}
		}
		//cv::imshow("plate_detect", gh);
		//cv::waitKey(0);
	}

    }
    return 0;
}
