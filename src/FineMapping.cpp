#include "FineMapping.h"
namespace prc{

    /*const int FINEMAPPING_H = 60;
    const int FINEMAPPING_W = 140;*/
	const int FINEMAPPING_H = 60;
	const int FINEMAPPING_W = 140;
    const int PADDING_UP_DOWN = 30;
    void drawRect(cv::Mat image,cv::Rect rect)
    {
        cv::Point p1(rect.x,rect.y);
        cv::Point p2(rect.x+rect.width,rect.y+rect.height);
        cv::rectangle(image,p1,p2,cv::Scalar(0,255,0),1);
    }


    FineMapping::FineMapping(std::string prototxt,std::string caffemodel) {
         net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);

    }

    cv::Mat FineMapping::FineMappingHorizon(cv::Mat FinedVertical,int leftPadding,int rightPadding)
    {

//        if(FinedVertical.channels()==1)
//            cv::cvtColor(FinedVertical,FinedVertical,cv::COLOR_GRAY2BGR);
        cv::Mat inputBlob = cv::dnn::blobFromImage(FinedVertical, 1/255.0, cv::Size(66,16),
                                      cv::Scalar(0,0,0),false);

        net.setInput(inputBlob,"data");
        cv::Mat prob = net.forward();
        int front = static_cast<int>(prob.at<float>(0,0)*FinedVertical.cols);
        int back = static_cast<int>(prob.at<float>(0,1)*FinedVertical.cols);
        front -= leftPadding ;
        if(front<0) front = 0;
        back +=rightPadding;
        if(back>FinedVertical.cols-1) back=FinedVertical.cols - 1;
        cv::Mat cropped  = FinedVertical.colRange(front,back).clone();
        return  cropped;


    }
    std::pair<int,int> FitLineRansac(std::vector<cv::Point> pts,int zeroadd = 0 )
    {
        std::pair<int,int> res;
        if(pts.size()>2)
        {
            cv::Vec4f line;
            cv::fitLine(pts,line,cv::DIST_HUBER,0,0.01,0.01);
            float vx = line[0];
            float vy = line[1];
            float x = line[2];
            float y = line[3];
            int lefty = static_cast<int>((-x * vy / vx) + y);
            int righty = static_cast<int>(((136- x) * vy / vx) + y);
            res.first = lefty+PADDING_UP_DOWN+zeroadd;
            res.second = righty+PADDING_UP_DOWN+zeroadd;
            return res;
        }
        res.first = zeroadd;
        res.second = zeroadd;
        return res;
    }

    cv::Mat FineMapping::FineMappingVertical(cv::Mat InputProposal,int sliceNum,int upper,int lower,int windows_size){
        cv::Mat PreInputProposal;
        cv::Mat proposal;
        cv::resize(InputProposal,PreInputProposal,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
        if(InputProposal.channels() == 3)
            cv::cvtColor(PreInputProposal,proposal,cv::COLOR_BGR2GRAY);
        else
            PreInputProposal.copyTo(proposal);
        // this will improve some sen
		//cv::imshow("op", PreInputProposal);
		//cv::waitKey(0);
        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,3));
        float diff = static_cast<float>(upper-lower);
        diff/=static_cast<float>(sliceNum-1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;
        int contours_nums=0;
        for(int i = 0 ; i < sliceNum ; i++)
        {
            std::vector<std::vector<cv::Point> > contours;
            float k =lower + i*diff;
            cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
            cv::Mat draw;
            binary_adaptive.copyTo(draw);
			//cv::imshow("op2", draw);
			//cv::waitKey(0);
            cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
			int width = binary_adaptive.rows;
			int height = binary_adaptive.cols;
			//std::cout << width << ":" << height << std::endl;
            for(auto contour: contours)
            {
                cv::Rect bdbox =cv::boundingRect(contour);
				
                float lwRatio = bdbox.height/static_cast<float>(bdbox.width);
                int  bdboxAera = bdbox.width*bdbox.height;
				//std::cout << "+++++++" << bdbox <<" "<< lwRatio<<" " <<bdboxAera <<std::endl;
                if ((   lwRatio>0.7&&bdbox.width*bdbox.height>100 && bdboxAera<300)
                    || (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
                {
                    cv::Point p1(bdbox.x, bdbox.y);
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
                    line_upper.push_back(p1);
                    line_lower.push_back(p2);
                    contours_nums+=1;
                }
            }
        }
		//std::cout << "^^^^^:" << contours_nums << std::endl;
        if(contours_nums<40)
        {
            cv::bitwise_not(InputProposal,InputProposal);
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,5));
            cv::Mat bak;
            cv::resize(InputProposal,bak,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
            cv::erode(bak,bak,kernal);
            if(InputProposal.channels() == 3)
                cv::cvtColor(bak,proposal,cv::COLOR_BGR2GRAY);
            else
                proposal = bak;
            int contours_nums=0;
            for(int i = 0 ; i < sliceNum ; i++)
            {
                std::vector<std::vector<cv::Point> > contours;
                float k =lower + i*diff;
                cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
                cv::Mat draw;
                binary_adaptive.copyTo(draw);
                cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                for(auto contour: contours)
                {
                    cv::Rect bdbox =cv::boundingRect(contour);
                    float lwRatio = bdbox.height/static_cast<float>(bdbox.width);
                    int  bdboxAera = bdbox.width*bdbox.height;
                    if ((   lwRatio>0.7&&bdbox.width*bdbox.height>120 && bdboxAera<300)
                        || (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
                    {

                        cv::Point p1(bdbox.x, bdbox.y);
                        cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
                        line_upper.push_back(p1);
                        line_lower.push_back(p2);
                        contours_nums+=1;
                    }
                }
            }
        }
            cv::Mat rgb;
            cv::copyMakeBorder(PreInputProposal, rgb, PADDING_UP_DOWN, PADDING_UP_DOWN, 0, 0, cv::BORDER_REPLICATE);
			
            std::pair<int, int> A;
            std::pair<int, int> B;
            A = FitLineRansac(line_upper, -1);
            B = FitLineRansac(line_lower, 1);
            int leftyB = A.first;
            int rightyB = A.second;
            int leftyA = B.first;
            int rightyA = B.second;
            int cols = rgb.cols;
            int rows = rgb.rows;
            std::vector<cv::Point2f> corners(4);
            corners[0] = cv::Point2f(cols - 10, rightyA-30);//第二个点，绿色
            corners[1] = cv::Point2f(0+10, leftyA-30);//第一个点，红色
            corners[2] = cv::Point2f(cols - 10, rightyB+30);//第三个点，蓝色
            corners[3] = cv::Point2f(0+10, leftyB+30);//第四个点，白色
            std::vector<cv::Point2f> corners_trans(4);
			/*circle(rgb, corners[0], 1, cv::Scalar(0, 255, 0), -1);
			circle(rgb, corners[1], 2, cv::Scalar(0, 0, 255), 1);
			circle(rgb, corners[2], 3, cv::Scalar(255, 0, 0), -1);
			circle(rgb, corners[3], 4, cv::Scalar(255, 255, 255), -1);*/
			//cv::imshow("op1", rgb);
			
			//int padding_x = cvFloor(rows * 0.01 * 5);
			//int padding_y = cvFloor(rows * 0.01 * 2);
			//corners_trans[0] = cv::Point2f(140 + padding_x, 32 + padding_y);
			//corners_trans[1] = cv::Point2f(0 + padding_x, 32 + padding_y);
			//corners_trans[2] = cv::Point2f(140 + padding_x, 0 + padding_y);
			//corners_trans[3] = cv::Point2f(0 + padding_x, 0 + padding_y);
   //       /*  corners_trans[0] = cv::Point2f(136+ padding_x, 36+ padding_y);
   //         corners_trans[1] = cv::Point2f(0+ padding_x, 36+ padding_y);
   //         corners_trans[2] = cv::Point2f(136+ padding_x, 0+ padding_y);
   //         corners_trans[3] = cv::Point2f(0+ padding_x, 0+ padding_y);*/
   //         cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
   //         //cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);
			//cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);

			int padding_x = cvFloor(cols * 0.00 * 5);
			int padding_y = cvFloor(rows * 0.00 * 2);
			corners_trans[0] = cv::Point2f(136 + padding_x, padding_y);//
			corners_trans[1] = cv::Point2f(0 + padding_x,   padding_y);
			corners_trans[2] = cv::Point2f(136 + padding_x, 36 + padding_y);
			corners_trans[3] = cv::Point2f(0 + padding_x, 36 + padding_y);
			
			
			/*  corners_trans[0] = cv::Point2f(136+ padding_x, 36+ padding_y);
			corners_trans[1] = cv::Point2f(0+ padding_x, 36+ padding_y);
			corners_trans[2] = cv::Point2f(136+ padding_x, 0+ padding_y);
			corners_trans[3] = cv::Point2f(0+ padding_x, 0+ padding_y);*/
			cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
			//cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);
			cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);


            cv::warpPerspective(rgb, quad, transform, quad.size());
        return quad;
    }
}


