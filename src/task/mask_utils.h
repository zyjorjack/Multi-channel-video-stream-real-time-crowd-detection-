#ifndef MASK_UTILS_H
#define MASK_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct CameraConfigInfo {
    std::string ip;
    std::string username;
    std::string password;
    int channel;
    int width;
    int height;
    std::vector<std::vector<cv::Point>> exclusion_zones;
};

std::vector<CameraConfigInfo> parseCameraConfig(const std::string& configFile);
cv::Mat createExclusionMask(int width, int height, const std::vector<std::vector<cv::Point>>& exclusion_zones);
uchar getMaskValueAtPoint(const cv::Point& p, const cv::Mat& mask);
bool shouldExcludeBox(const cv::Rect& box, const cv::Mat& mask);

#endif // MASK_UTILS_H