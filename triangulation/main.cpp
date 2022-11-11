#include <chrono>
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

int main(){
    
    std::string path = "/Users/anshijie/debug/mesh/0.png";
    cv::Mat mask = cv::imread(path, 0);
    
    cv::Rect rect = cv::boundingRect(mask);
    
    cv::Subdiv2D subdiv(rect);
    
    int rows = mask.rows, cols = mask.cols;
    
    for (int y = 0; y < rows; y++){
        for (int x = 0; x < cols; x++){
            if (mask.at<uint8_t>(y,x) > 0){
                cv::Point2f p(x*1.0, y*1.0);
                subdiv.insert(p);
            }
        }
    }
    
    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);
    
    cv::Mat map = cv::Mat::zeros(rows*4, cols*4, CV_8UC3);
    
    for (auto vec : triangles){
        cv::line(map, cv::Point2i(vec[0]*4, vec[1]*4), cv::Point2i(vec[2]*4, vec[3]*4), cv::Scalar(255, 255, 255));
        cv::line(map, cv::Point2i(vec[2]*4, vec[3]*4), cv::Point2i(vec[4]*4, vec[5]*4), cv::Scalar(255, 255, 255));
        cv::line(map, cv::Point2i(vec[0]*4, vec[1]*4), cv::Point2i(vec[4]*4, vec[5]*4), cv::Scalar(255, 255, 255));
    }
    
    cv::imwrite("/Users/anshijie/debug/mesh/0triangle3.3.png", map);
    
    return 0;
}
