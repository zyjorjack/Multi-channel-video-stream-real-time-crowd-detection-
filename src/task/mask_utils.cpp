#include "mask_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <opencv2/imgcodecs.hpp> // 添加头文件用于保存图像

std::vector<CameraConfigInfo> parseCameraConfig(const std::string& configFile) {
    std::vector<CameraConfigInfo> configs;
    std::ifstream file(configFile);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        CameraConfigInfo config;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 4) continue;
        
        config.ip = tokens[0];
        config.username = tokens[1];
        config.password = tokens[2];
        config.channel = std::stoi(tokens[3]);
        
        // 设置默认分辨率
        config.width = 1920;
        config.height = 1080;
        
        // 解析分辨率
        if (tokens.size() > 4 && tokens[4].find('*') != std::string::npos) {
            size_t pos = tokens[4].find('*');
            try {
                config.width = std::stoi(tokens[4].substr(0, pos));
                config.height = std::stoi(tokens[4].substr(pos+1));
            } catch (...) {
                // 保持默认值
            }
        }
        
        // 解析多边形
        size_t poly_start = (tokens.size() > 4 && tokens[4].find('*') != std::string::npos) ? 5 : 4;
        
        for (size_t i = poly_start; i < tokens.size(); i++) {
            if (tokens[i][0] == '[') {
                std::string poly_str;
                size_t end_pos = tokens[i].find(']');
                if (end_pos != std::string::npos) {
                    poly_str = tokens[i].substr(1, end_pos-1);
                } else {
                    poly_str = tokens[i].substr(1);
                    for (i++; i < tokens.size(); i++) {
                        end_pos = tokens[i].find(']');
                        if (end_pos != std::string::npos) {
                            poly_str += " " + tokens[i].substr(0, end_pos);
                            break;
                        }
                        poly_str += " " + tokens[i];
                    }
                }

                std::vector<cv::Point> polygon;
                std::istringstream point_ss(poly_str);
                std::string point;
                while (point_ss >> point) {
                    size_t comma_pos = point.find(',');
                    if (comma_pos != std::string::npos) {
                        try {
                            int x = std::stoi(point.substr(0, comma_pos));
                            int y = std::stoi(point.substr(comma_pos+1));
                            x = std::max(0, std::min(x, config.width-1));
                            y = std::max(0, std::min(y, config.height-1));
                            polygon.emplace_back(x, y);
                        } catch (...) {
                            // 忽略无效坐标
                        }
                    }
                }

                if (polygon.size() >= 3) {
                    if (polygon.front() != polygon.back()) {
                        polygon.push_back(polygon.front());
                    }
                    config.exclusion_zones.push_back(polygon);
                }
            }
        }
        
        // ===== 新增：生成并保存掩膜 =====
        cv::Mat mask = createExclusionMask(config.width, config.height, config.exclusion_zones);
        // 生成文件名（IP中的冒号替换为下划线）
        std::string safe_ip = config.ip;
        std::replace(safe_ip.begin(), safe_ip.end(), ':', '_');
        std::string filename = "mask_" + safe_ip + "_ch" + std::to_string(config.channel) + ".png";
        
        // 保存掩膜图像
        if (!mask.empty()) {
            if (cv::imwrite(filename, mask)) {
                std::cout << "Saved mask for camera " << config.ip 
                          << " (channel " << config.channel << ") to: " 
                          << filename << std::endl;
            } else {
                std::cerr << "Error: Failed to save mask for camera " << config.ip 
                          << " (channel " << config.channel << ")" << std::endl;
            }
        }
        // ===== 新增结束 =====
        
        configs.push_back(config);
    }
    
    return configs;
}

uchar getMaskValueAtPoint(const cv::Point& p, const cv::Mat& mask) {
    if (p.x < 0 || p.y < 0 || p.x >= mask.cols || p.y >= mask.rows) {
        return 0;
    }
    return mask.at<uchar>(p);
}

cv::Mat createExclusionMask(int width, int height, const std::vector<std::vector<cv::Point>>& exclusion_zones) {
    // 创建全黑掩膜（黑色=检测区域）
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

    // 没有排除区域直接返回
    if (exclusion_zones.empty()) {
        return mask;
    }

    // 计算灰度级步长
    const int n = exclusion_zones.size();
    const float step = n > 1 ? (154.0f / (n - 1)) : 0;

    // 填充排除区域
    for (size_t z = 0; z < exclusion_zones.size(); ++z) {
        const auto& zone = exclusion_zones[z];
        if (zone.size() < 3) continue;

        // 计算灰度值
        uchar gray_value = n == 1 ? 255 : static_cast<uchar>(255 - z * step);
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{zone}, cv::Scalar(gray_value), 8);
    }
    
    return mask;
}

bool shouldExcludeBox(const cv::Rect& box, const cv::Mat& mask) {
    // 计算中心点
    cv::Point center(box.x + box.width/2, box.y + box.height/2);
    uchar centerValue = getMaskValueAtPoint(center, mask);
    
    // 中心点在检测区域
    if (centerValue == 0) {
        return false;
    }
    
    // 检查四个角点
    const std::vector<cv::Point> corners = {
        cv::Point(box.x + 1, box.y + 1),
        cv::Point(box.x + box.width - 1, box.y + 1),
        cv::Point(box.x + 1, box.y + box.height - 1),
        cv::Point(box.x + box.width - 1, box.y + box.height - 1)
    };
    
    for (const auto& corner : corners) {
        if (getMaskValueAtPoint(corner, mask) != centerValue) {
            return false;
        }
    }
    
    return true;
}