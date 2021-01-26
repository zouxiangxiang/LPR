//
// Created by Jack Yu on 23/10/2017.
//

#include "Pipeline.h"

namespace prc{



    const int HorizontalPadding = 2;
	PipelinePR::PipelinePR(std::string detector_filename,
                           std::string finemapping_prototxt, std::string finemapping_caffemodel,
                           std::string segmentation_prototxt, std::string segmentation_caffemodel,
                           std::string charRecognization_proto, std::string charRecognization_caffemodel,
                           std::string segmentationfree_proto,std::string segmentationfree_caffemodel) {
        plateDetection = new PlateDetection(detector_filename);
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
        plateSegmentation = new PlateSegmentation(segmentation_prototxt, segmentation_caffemodel);
        generalRecognizer = new CNNRecognizer(charRecognization_proto, charRecognization_caffemodel);
        segmentationFreeRecognizer =  new SegmentationFreeRecognizer(segmentationfree_proto,segmentationfree_caffemodel);

    }

    PipelinePR::~PipelinePR() {

       // delete plateDetection;
       /* delete fineMapping;
        delete plateSegmentation;
        delete generalRecognizer;
        delete segmentationFreeRecognizer;*/


    }

    std::vector<PlateInfo_> PipelinePR:: RunPiplineAsImage(cv::Mat plateImage,int method) {
        std::vector<PlateInfo_> results;
        std::vector<prc::PlateInfo_> plates;
       // plateDetection->plateDetectionRough(plateImage,plates,36,700);
		prc::PlateInfo_  plateinfo;
       // for (pr::PlateInfo plateinfo:plates) {
		cv::Mat image_finemapping = plateImage;
            //cv::Mat image_finemapping = plateinfo.getPlateImage();
           // image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
			
          //  image_finemapping = pr::fastdeskew(image_finemapping, 5);
			/*cv::imshow("name", image_finemapping);
			cv::waitKey(1);
*/
            //Segmentation-based
		std::cout << "+++++++++++++++++++++" << std::endl;
            if(method==SEGMENTATION_BASED_METHOD)
            {
                image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 2, HorizontalPadding);
                cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
				imshow(" image_finemapping", image_finemapping);
				cv::waitKey(0);
                plateinfo.setPlateImage(image_finemapping);
                std::vector<cv::Rect> rects;
                plateSegmentation->segmentPlatePipline(plateinfo, 1, rects);
                plateSegmentation->ExtractRegions(plateinfo, rects);
                cv::copyMakeBorder(image_finemapping, image_finemapping, 0, 0, 0, 20, cv::BORDER_REPLICATE);
                plateinfo.setPlateImage(image_finemapping);
                generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
                plateinfo.decodePlateNormal(prc::CH_PLATE_CODE);

            }
                //Segmentation-free
            else if(method==SEGMENTATION_FREE_METHOD)
            {
				//imshow(" image_finemapping_", image_finemapping);
				
                //image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 6, HorizontalPadding+12);
				//image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding + 3);
               // cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
				//image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
				//image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
				//image_finemapping = prc::fastdeskew(image_finemapping, 5);
				cv::resize(image_finemapping, image_finemapping, cv::Size(130 , 36));
                plateinfo.setPlateImage(image_finemapping);
				//imshow(" image_finemapping", image_finemapping);
				//cv::waitKey(0);

                std::pair<std::string,float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(),prc::CH_PLATE_CODE);
				std::cout << "++++++++++++++++" << res.first << std::endl ;
                plateinfo.confidence = res.second;
                plateinfo.setPlateName(res.first);
            }
            results.push_back(plateinfo);
       // }

        return results;

    }//namespace pr

	
	std::vector<PlateInfo_> PipelinePR::RunPiplineAsImage_(cv::Mat plateImage, int method) {
		std::vector<PlateInfo_> results;
		std::vector<prc::PlateInfo_> plates;
		 plateDetection->plateDetectionRough(plateImage,plates,36,700);
		 std::cout << "+++++++++uuu++++++++++++" << std::endl;
		 for (prc::PlateInfo_ plateinfo:plates) {
		
		cv::Mat image_finemapping = plateinfo.getPlateImage();
		//imshow(" image_finemapping", image_finemapping);
		//cv::waitKey(0);

		image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
		imshow(" FineMappingVertical", image_finemapping);
		cv::waitKey(0);
		image_finemapping = prc::fastdeskew(image_finemapping, 5);
		//imshow(" fastdeskew", image_finemapping);
		//cv::waitKey(0);


		//Segmentation-based

		if (method == SEGMENTATION_BASED_METHOD)
		{
			image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 2, HorizontalPadding);
			cv::resize(image_finemapping, image_finemapping, cv::Size(136 + HorizontalPadding, 36));
			plateinfo.setPlateImage(image_finemapping);
			std::vector<cv::Rect> rects;
			plateSegmentation->segmentPlatePipline(plateinfo, 1, rects);
			plateSegmentation->ExtractRegions(plateinfo, rects);
			cv::copyMakeBorder(image_finemapping, image_finemapping, 0, 0, 0, 20, cv::BORDER_REPLICATE);
			plateinfo.setPlateImage(image_finemapping);
			generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
			plateinfo.decodePlateNormal(prc::CH_PLATE_CODE);

		}
		//Segmentation-free
		else if (method == SEGMENTATION_FREE_METHOD)
		{
			image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding + 3);
			cv::resize(image_finemapping, image_finemapping, cv::Size(136 + HorizontalPadding, 36));
			plateinfo.setPlateImage(image_finemapping);

			std::pair<std::string, float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(), prc::CH_PLATE_CODE);
			plateinfo.confidence = res.second;
			plateinfo.setPlateName(res.first);
		}
		results.push_back(plateinfo);
		 }

		return results;

	}//namespace pr


}
