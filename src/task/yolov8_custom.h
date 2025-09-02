#ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"
#include "types/error.h"  // 修改为正确的错误码头文件路径
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

class Yolov8Custom {
public:
    Yolov8Custom();
    ~Yolov8Custom();

    // 禁止复制
    Yolov8Custom(const Yolov8Custom&) = delete;
    Yolov8Custom& operator=(const Yolov8Custom&) = delete;

    nn_error_e LoadModel(const char *model_path);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);

private:
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects);
    void LetterboxDecode(std::vector<Detection> &objects, bool hor, int pad);

    bool ready_;
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool want_float_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    std::shared_ptr<NNEngine> engine_;
    std::mutex model_mutex_;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H