#ifndef RK3588_DEMO_Yolov8_THREAD_POOL_H
#define RK3588_DEMO_Yolov8_THREAD_POOL_H

#include "yolov8_custom.h"
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>

#include <sqlite3.h>
#include <ctime>

// 先定义FrameResultQueue类
class FrameResultQueue {
private:
    std::deque<std::pair<int, int>> queue;  // <frame_id, box_count>
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stop_flag{false};

public:
    void push(int frame_id, int box_count) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.emplace_back(frame_id, box_count);
        cv.notify_one();
    }

    bool pop(int& frame_id, int& box_count, int timeout_ms = 100) {
    std::unique_lock<std::mutex> lock(mtx);
    if (!cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
        [this]{ return !queue.empty() || stop_flag; })) {
        return false; // 超时返回
    }
    
    if (stop_flag && queue.empty()) return false;
    
    auto result = queue.front();
    queue.pop_front();
    frame_id = result.first;
    box_count = result.second;
    return true;
   }

    void stop() {
        stop_flag = true;
        cv.notify_all();
    }
};

class Yolov8ThreadPool
{
private:
    std::queue<std::pair<int, cv::Mat>> tasks;
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;
    std::map<int, std::vector<Detection>> results;
    std::map<int, cv::Mat> img_results;
    std::vector<std::thread> threads;
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;

    FrameResultQueue frame_results;  // 现在可以正确使用了
    std::atomic<bool> processing_complete{false}; // 新增标志位
    std::atomic<int> submitted_frames{0};
    std::atomic<int> processed_frames{0};

    bool stop;

    void worker(int id);

public:
    Yolov8ThreadPool();
    ~Yolov8ThreadPool();

    nn_error_e setUp(const std::string &model_path, int num_threads = 12);
    nn_error_e submitTask(const cv::Mat &img, int id);
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id);
    nn_error_e getTargetImgResult(cv::Mat &img, int id);
    nn_error_e getTargetImgResultWithCount(cv::Mat &img, int id, int& box_count);
    // 添加新方法声明
    nn_error_e getTargetImgResultWithDetections(cv::Mat& img, int id, int& box_count, std::vector<Detection>& detections);

    bool allTasksDone() const;
    int getSubmittedCount() const { return submitted_frames; }
    int getProcessedCount() const { return processed_frames; }

    void setProcessingComplete() { processing_complete = true; }
    bool isProcessingComplete() const { return processing_complete; }

    void pushFrameResult(int frame_id, int box_count) {
        frame_results.push(frame_id, box_count);
    }
    
    bool popFrameResult(int& frame_id, int& box_count) {
        return frame_results.pop(frame_id, box_count);
    }
    
    void stopResultQueue() {
        frame_results.stop();
    }

    void stopAll();
};

#endif // RK3588_DEMO_Yolov8_THREAD_POOL_H