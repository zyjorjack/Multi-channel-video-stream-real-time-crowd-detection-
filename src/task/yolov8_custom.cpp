#include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static std::vector<std::string> g_classes = {"person"};

Yolov8Custom::Yolov8Custom() {
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr) {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_) {
        if (tensor.data != nullptr) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

nn_error_e Yolov8Custom::LoadModel(const char *model_path) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS) {
        NN_LOG_ERROR("yolov8 load model file failed");
        return NN_LOAD_MODEL_FAIL;
    }

    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1) {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }

    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);
    if (!input_tensor_.data) {
        NN_LOG_ERROR("Failed to allocate input tensor memory");
        return NN_RKNN_INPUT_SET_FAIL;
    }

    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 6) {
        NN_LOG_ERROR("yolov8 output tensor number is not 6, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    want_float_ = (output_shapes[0].type == NN_TENSOR_FLOAT16);
    if (want_float_) {
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }

    output_tensors_.clear();
    out_zps_.clear();
    out_scales_.clear();
    
    for (int i = 0; i < output_shapes.size(); i++) {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++) {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        if (!tensor.data) {
            NN_LOG_ERROR("Failed to allocate output tensor memory");
            return NN_RKNN_OUTPUT_GET_FAIL;
        }
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }

    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox) {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;

    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    if (process_type == "opencv") {
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    } 
    else if (process_type == "rga") {
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        cvimg2tensor_rga(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    else {
        return NN_RKNN_INPUT_ATTR_ERROR;
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference() {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;

    std::lock_guard<std::mutex> lock(model_mutex_);
    std::vector<tensor_data_s> inputs = {input_tensor_};
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects) {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;

    std::lock_guard<std::mutex> lock(model_mutex_);
    void *output_data[6];
    for (int i = 0; i < 6; i++) {
        output_data[i] = output_tensors_[i].data;
    }

    std::vector<float> DetectiontRects;
    if (want_float_) {
        yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
    } else {
        yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
    }

    int img_width = img.cols;
    int img_height = img.rows;
    objects.clear();

    for (size_t i = 0; i < DetectiontRects.size(); i += 6) {
        if (i + 5 >= DetectiontRects.size()) break;

        int classId = static_cast<int>(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        if (conf < 0.25f) continue;

        int xmin = static_cast<int>(DetectiontRects[i + 2] * img_width + 0.5f);
        int ymin = static_cast<int>(DetectiontRects[i + 3] * img_height + 0.5f);
        int xmax = static_cast<int>(DetectiontRects[i + 4] * img_width + 0.5f);
        int ymax = static_cast<int>(DetectiontRects[i + 5] * img_height + 0.5f);

        xmin = std::max(0, std::min(xmin, img_width - 1));
        ymin = std::max(0, std::min(ymin, img_height - 1));
        xmax = std::max(0, std::min(xmax, img_width - 1));
        ymax = std::max(0, std::min(ymax, img_height - 1));

        if (xmax <= xmin || ymax <= ymin) continue;


        Detection result;
        result.class_id = classId;
        result.confidence = conf;
        result.color = cv::Scalar(0, 255, 0);
        result.className = classId < g_classes.size() ? g_classes[classId] : "unknown";

        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return NN_SUCCESS;
}

void Yolov8Custom::LetterboxDecode(std::vector<Detection> &objects, bool hor, int pad) {
    for (auto &obj : objects) {
        if (hor) {
            obj.box.x -= pad;
        } else {
            obj.box.y -= pad;
        }
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects) {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;

    cv::Mat image_letterbox;
    auto ret = Preprocess(img, "opencv", image_letterbox);
    if (ret != NN_SUCCESS) return ret;

    ret = Inference();
    if (ret != NN_SUCCESS) return ret;

    ret = Postprocess(image_letterbox, objects);
    if (ret != NN_SUCCESS) return ret;

    LetterboxDecode(objects, letterbox_info_.hor, letterbox_info_.pad);
    return NN_SUCCESS;
}