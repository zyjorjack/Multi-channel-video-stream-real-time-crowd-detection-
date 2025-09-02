#include <iostream>
#include <csignal>
#include "HCNetSDK.h"
#include "LinuxPlayM4.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <thread>
#include <sstream>
#include <memory>
#include <mutex>
#include <atomic>
#include <sqlite3.h>
#include <chrono>
#include <iomanip>
#include "task/yolov8_thread_pool.h"
#include "task/mask_utils.h"
#include "task/comm.h"
#include <X11/Xlib.h>
#include <unordered_map>

using namespace cv;

// Global mutex lock
std::mutex g_gui_mutex;
std::atomic<bool> g_running{true};

// Database connection pool
namespace {
    std::mutex db_pool_mutex;
    std::map<std::string, sqlite3*> db_pool;
}

// Hikvision SDK mutex lock
std::mutex g_hik_mutex;
std::atomic<uint16_t> g_max_box_count(0);

// Signal handler function
void signal_handler(int signal) {
    g_running = false;
}

// Camera configuration structure
struct CameraConfig {
    std::string unique_id;  // Unique identifier
    std::string ip;
    std::string username;
    std::string password;
    int channel;
    LONG userID = -1;
    LONG realPlayHandle = -1;
    std::unique_ptr<Yolov8ThreadPool> yolov8_pool;
    std::atomic<int> frame_id{0};
    std::atomic<bool> stop_flag{false};
    sqlite3* db = nullptr;
    sqlite3* send_db = nullptr;
    time_t last_minute = 0;
    
    // Video stream related
    int g_nPort = -1;
    Mat g_BGRImage;
    std::mutex g_frame_mutex;
    
    // Performance statistics
    std::atomic<int> frame_counter{0};
    std::chrono::steady_clock::time_point last_stat_time;
    double fps{0};

    cv::Mat exclusion_mask;
    std::mutex mask_mutex;
    
    CameraConfig(const CameraConfig&) = delete;
    CameraConfig& operator=(const CameraConfig&) = delete;
    
    CameraConfig(CameraConfig&& other) noexcept
        : unique_id(std::move(other.unique_id)),
          ip(std::move(other.ip)),
          username(std::move(other.username)),
          password(std::move(other.password)),
          channel(other.channel),
          userID(other.userID),
          realPlayHandle(other.realPlayHandle),
          yolov8_pool(std::move(other.yolov8_pool)),
          frame_id(other.frame_id.load()),
          stop_flag(other.stop_flag.load()),
          db(other.db),
          send_db(other.send_db),
          last_minute(other.last_minute),
          g_nPort(other.g_nPort),
          g_BGRImage(std::move(other.g_BGRImage)),
          frame_counter(other.frame_counter.load()),
          last_stat_time(other.last_stat_time),
          fps(other.fps),
          exclusion_mask(std::move(other.exclusion_mask))
    {
        other.db = nullptr;
        other.send_db = nullptr;
        other.g_nPort = -1;
        other.frame_counter = 0;
    }
    
    CameraConfig& operator=(CameraConfig&& other) noexcept {
        if (this != &other) {
            unique_id = std::move(other.unique_id);
            ip = std::move(other.ip);
            username = std::move(other.username);
            password = std::move(other.password);
            channel = other.channel;
            userID = other.userID;
            realPlayHandle = other.realPlayHandle;
            yolov8_pool = std::move(other.yolov8_pool);
            frame_id = other.frame_id.load();
            stop_flag = other.stop_flag.load();
            db = other.db;
            send_db = other.send_db;
            last_minute = other.last_minute;
            g_nPort = other.g_nPort;
            g_BGRImage = std::move(other.g_BGRImage);
            frame_counter = other.frame_counter.load();
            last_stat_time = other.last_stat_time;
            fps = other.fps;
            exclusion_mask = std::move(other.exclusion_mask);

            other.db = nullptr;
            other.send_db = nullptr;
            other.g_nPort = -1;
            other.frame_counter = 0;
        }
        return *this;
    }
    
    CameraConfig() = default;
    
    ~CameraConfig() {
        stop_flag = true;
        if (db) sqlite3_close(db);
        if (send_db) sqlite3_close(send_db);
        if (g_nPort != -1) {
            PlayM4_Stop(g_nPort);
            PlayM4_CloseStream(g_nPort);
            PlayM4_FreePort(g_nPort);
        }
    }
};

// Global configuration
std::string g_model_path;
int g_num_threads_per_camera = 2;
const int MAX_CAMERAS = 4;

// Fixed callback function signature - added nReserved2 parameter
void CALLBACK DecCBFun(int nPort, char* pBuf, int nSize, FRAME_INFO* pFrameInfo, void* nUser, int nReserved2) {
    CameraConfig* config = reinterpret_cast<CameraConfig*>(nUser);
    if (pFrameInfo->nType == T_YV12 && !config->stop_flag) {
        std::unique_lock<std::mutex> lock(config->g_frame_mutex, std::try_to_lock);
        if (lock.owns_lock()) {
            Mat yuvImg(pFrameInfo->nHeight + pFrameInfo->nHeight/2, pFrameInfo->nWidth, CV_8UC1, (uchar*)pBuf);
            cvtColor(yuvImg, config->g_BGRImage, COLOR_YUV2BGR_YV12);
        }
    }
}

// Real-time data callback
void CALLBACK RealDataCallBack_V30(LONG lPlayHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void* pUser) {
    CameraConfig* config = static_cast<CameraConfig*>(pUser);
    if (dwDataType == NET_DVR_STREAMDATA && dwBufSize > 0 && config->g_nPort != -1 && !config->stop_flag) {
        if (!PlayM4_InputData(config->g_nPort, pBuffer, dwBufSize)) {
            std::cerr << "PlayM4 input data failed: " << NET_DVR_GetLastError() << std::endl;
        }
    }
}

// Get timestamp
std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// Initialize database
sqlite3* GetDatabaseConnection(const std::string& db_name) {
    std::lock_guard<std::mutex> lock(db_pool_mutex);
    
    if (db_pool.find(db_name) != db_pool.end()) {
        return db_pool[db_name];
    }

    std::string db_path = "./" + db_name;
    sqlite3* db = nullptr;
    
    system(("mkdir -p $(dirname '" + db_path + "') 2>/dev/null").c_str());

    if (sqlite3_open_v2(db_path.c_str(), &db, 
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, 
        nullptr) != SQLITE_OK) 
    {
        std::cerr << "Database connection failed: " << sqlite3_errmsg(db) << std::endl;
        return nullptr;
    }

    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);

    const char* create_sql = "CREATE TABLE IF NOT EXISTS detection_results ("
                           "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                           "device TEXT NOT NULL,"
                           "timestamp TEXT NOT NULL,"
                           "box_count INTEGER NOT NULL);";
    
    char* errMsg = nullptr;
    if (sqlite3_exec(db, create_sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Create table failed: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        sqlite3_close(db);
        return nullptr;
    }

    db_pool[db_name] = db;
    return db;
}

// Save to database
bool SaveToDatabase(sqlite3* db, const std::string& device, int boxCount) {
    if (boxCount <= 0) return true;

    const char* sql = "INSERT INTO detection_results (device, timestamp, box_count) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "SQL preparation failed: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }
    
    std::string timestamp = GetCurrentTimestamp();
    sqlite3_bind_text(stmt, 1, device.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, boxCount);
    
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    
    if (success && boxCount > 0) {
        uint16_t current_max = g_max_box_count.load();
        if (boxCount > current_max) {
            g_max_box_count.store(boxCount);
            update_max_info(device, timestamp);
            send_people_count(boxCount);
        }
    }
    return success;
}

std::vector<CameraConfig> ReadCameraConfig(const std::string& configFile) {
    std::cout << "=== Reading camera configuration ===" << std::endl;
    auto configs = parseCameraConfig(configFile);
    std::vector<CameraConfig> cameras;
    
    // Counter for tracking same IP and channel
    std::unordered_map<std::string, int> camera_counter;
    
    for (const auto& cfg : configs) {
        CameraConfig camera;
        camera.ip = cfg.ip;
        camera.username = cfg.username;
        camera.password = cfg.password;
        camera.channel = cfg.channel;
        
        // Create exclusion mask
        camera.exclusion_mask = createExclusionMask(cfg.width, cfg.height, cfg.exclusion_zones);
        
        // Generate unique ID: IP + Channel + counter
        std::string base_id = camera.ip + "_Ch" + std::to_string(camera.channel);
        
        // Update counter
        int count = ++camera_counter[base_id];
        camera.unique_id = base_id + "_" + std::to_string(count);
        
        cameras.emplace_back(std::move(camera));
    }
    
    return cameras;
}

// Camera processing thread
void ProcessCameraStream(CameraConfig& cameraConfig) {
    // Initialize database
    cameraConfig.db = GetDatabaseConnection("detection_results.db");
    cameraConfig.send_db = GetDatabaseConnection("detection_results_send.db");

    if (!cameraConfig.db || !cameraConfig.send_db) {
        std::cerr << "Fatal error: Failed to initialize database connection" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize YOLOv8 thread pool
    cameraConfig.yolov8_pool = std::make_unique<Yolov8ThreadPool>();
    if (cameraConfig.yolov8_pool->setUp(g_model_path, g_num_threads_per_camera) != NN_SUCCESS) {
        std::cerr << "Failed to initialize YOLOv8 thread pool: " << cameraConfig.ip << std::endl;
        return;
    }

    // Device login
    NET_DVR_USER_LOGIN_INFO loginInfo = {0};
    NET_DVR_DEVICEINFO_V40 deviceInfo = {0};
    
    strncpy(loginInfo.sDeviceAddress, cameraConfig.ip.c_str(), sizeof(loginInfo.sDeviceAddress)-1);
    loginInfo.wPort = 8000;
    strncpy(loginInfo.sUserName, cameraConfig.username.c_str(), sizeof(loginInfo.sUserName)-1);
    strncpy(loginInfo.sPassword, cameraConfig.password.c_str(), sizeof(loginInfo.sPassword)-1);

    {
        std::lock_guard<std::mutex> lock(g_hik_mutex);
        cameraConfig.userID = NET_DVR_Login_V40(&loginInfo, &deviceInfo);
    }
    
    if (cameraConfig.userID < 0) {
        std::cerr << "Login failed: " << cameraConfig.ip << " Error: " << NET_DVR_GetLastError() << std::endl;
        return;
    }

    // Initialize playback library
    if (!PlayM4_GetPort(&cameraConfig.g_nPort)) {
        std::cerr << "Failed to get playback port: " << cameraConfig.ip << std::endl;
        NET_DVR_Logout(cameraConfig.userID);
        return;
    }

    if (!PlayM4_SetStreamOpenMode(cameraConfig.g_nPort, STREAME_REALTIME)) {
        std::cerr << "Failed to set stream mode: " << cameraConfig.ip << std::endl;
        NET_DVR_Logout(cameraConfig.userID);
        return;
    }

    if (!PlayM4_OpenStream(cameraConfig.g_nPort, NULL, 0, 1024*1024)) {
        std::cerr << "Failed to open stream: " << cameraConfig.ip << std::endl;
        NET_DVR_Logout(cameraConfig.userID);
        return;
    }

    // Use fixed callback function signature
    if (!PlayM4_SetDecCallBackExMend(cameraConfig.g_nPort, DecCBFun, NULL, 0, &cameraConfig)) {
        std::cerr << "Failed to set decode callback: " << cameraConfig.ip << std::endl;
        NET_DVR_Logout(cameraConfig.userID);
        return;
    }

    if (!PlayM4_Play(cameraConfig.g_nPort, 0)) {
        std::cerr << "Failed to start playback: " << cameraConfig.ip << std::endl;
        NET_DVR_Logout(cameraConfig.userID);
        return;
    }

    // Set preview parameters
    NET_DVR_PREVIEWINFO previewInfo = {0};
    previewInfo.lChannel = cameraConfig.channel;
    previewInfo.dwStreamType = 0;
    previewInfo.dwLinkMode = 0;
    previewInfo.bBlocked = 1;

    // Start preview
    {
        std::lock_guard<std::mutex> lock(g_hik_mutex);
        cameraConfig.realPlayHandle = NET_DVR_RealPlay_V40(cameraConfig.userID, &previewInfo, RealDataCallBack_V30, &cameraConfig);
    }
    
    if (cameraConfig.realPlayHandle < 0) {
        std::cerr << "Failed to start preview: " << cameraConfig.ip << " Error: " << NET_DVR_GetLastError() << std::endl;
        NET_DVR_Logout(cameraConfig.userID);
        return;
    }

    // Create display window - use unique ID to ensure unique window name
    std::string windowName = "Camera " + cameraConfig.unique_id;
    {
        std::lock_guard<std::mutex> gui_lock(g_gui_mutex);
        cv::namedWindow(windowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(windowName, 640, 360);
    }

    cameraConfig.last_stat_time = std::chrono::steady_clock::now();

    // Main processing loop
    while (!cameraConfig.stop_flag && g_running) {
        // Reset frame_id every minute
        time_t current_time = time(nullptr);
        if (localtime(&current_time)->tm_min != localtime(&cameraConfig.last_minute)->tm_min) {
            cameraConfig.frame_id.store(1);
            cameraConfig.last_minute = current_time;
        }

        // Get video frame
        cv::Mat frameCopy;
        {
            std::unique_lock<std::mutex> lock(cameraConfig.g_frame_mutex, std::try_to_lock);
            if (lock.owns_lock() && !cameraConfig.g_BGRImage.empty()) {
                frameCopy = cameraConfig.g_BGRImage.clone();
            }
        }

        if (!frameCopy.empty()) {
            int currentFrameId = cameraConfig.frame_id++;
            cameraConfig.yolov8_pool->submitTask(frameCopy, currentFrameId);

            // Get inference results
            cv::Mat resultImg;
            int rawBoxCount = 0;
            std::vector<Detection> detections;
            
            if (cameraConfig.yolov8_pool->getTargetImgResultWithDetections(
                resultImg, currentFrameId, rawBoxCount, detections) == NN_SUCCESS) {
                
                // Filter detection boxes
                int filteredBoxCount = 0;
                {
                    std::lock_guard<std::mutex> mask_lock(cameraConfig.mask_mutex);
                    for (const auto& det : detections) {
                        cv::Rect safeBox = det.box;
                        safeBox.x = std::max(0, std::min(safeBox.x, resultImg.cols - 1));
                        safeBox.y = std::max(0, std::min(safeBox.y, resultImg.rows - 1));
                        safeBox.width = std::min(safeBox.width, resultImg.cols - safeBox.x);
                        safeBox.height = std::min(safeBox.height, resultImg.rows - safeBox.y);
                        
                        if (safeBox.width <= 0 || safeBox.height <= 0) continue;

                        // da ying
                        // std::cerr << safeBox << std::endl;

                        if (shouldExcludeBox(safeBox, cameraConfig.exclusion_mask)) {
                            cv::rectangle(resultImg, safeBox, cv::Scalar(0, 0, 255), 2);
                        } else {
                            cv::rectangle(resultImg, safeBox, cv::Scalar(0, 255, 0), 2);
                            filteredBoxCount++;
                        }
                    }
                }

                // Display information
                std::string infoText = cameraConfig.unique_id;
                cv::putText(resultImg, infoText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                            0.7, cv::Scalar(0, 255, 0), 2);
                
                std::string countText = "Valid count: " + std::to_string(filteredBoxCount) + 
                                       " (Raw: " + std::to_string(rawBoxCount) + ")";
                cv::putText(resultImg, countText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 
                            0.7, cv::Scalar(0, 255, 0), 2);

                // Display results
                {
                    std::lock_guard<std::mutex> gui_lock(g_gui_mutex);
                    cv::imshow(windowName, resultImg);
                }
                
                // Save results to database
                if (!SaveToDatabase(cameraConfig.db, cameraConfig.unique_id, filteredBoxCount)) {
                    std::cerr << "Failed to save to database: " << cameraConfig.unique_id << std::endl;
                }
            }
        }

        // Performance statistics
        if (cameraConfig.frame_counter++ % 30 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - cameraConfig.last_stat_time).count() / 1000.0;
            if (elapsed > 0) {
                cameraConfig.fps = 30 / elapsed;
                cameraConfig.last_stat_time = now;
                std::cout << "Camera " << cameraConfig.unique_id << " FPS: " << cameraConfig.fps << std::endl;
            }
        }

        // Handle exit event
        {
            std::lock_guard<std::mutex> gui_lock(g_gui_mutex);
            if (cv::waitKey(1) == 27) {
                cameraConfig.stop_flag = true;
                g_running = false;
            }
        }

        // Dynamic delay adjustment
        static auto last_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        
        int delay = std::max(5, 30 - static_cast<int>(elapsed));
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    // Cleanup resources
    {
        std::lock_guard<std::mutex> lock(g_hik_mutex);
        if (cameraConfig.realPlayHandle >= 0) {
            NET_DVR_StopRealPlay(cameraConfig.realPlayHandle);
        }
        if (cameraConfig.userID >= 0) {
            NET_DVR_Logout(cameraConfig.userID);
        }
    }
    
    {
        std::lock_guard<std::mutex> gui_lock(g_gui_mutex);
        cv::destroyWindow(windowName);
    }
}

int main(int argc, char** argv) {
    // Initialize X11 thread support
    XInitThreads();

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Set DISPLAY environment variable
    setenv("DISPLAY", ":0", 1);

    // Parameter check
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <config_file> [threads_per_camera]" << std::endl;
        return -1;
    }

    g_model_path = argv[1];
    std::string configFile = argv[2];
    
    // Set thread count
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (argc > 3) {
        g_num_threads_per_camera = std::max(1, std::min(atoi(argv[3]), static_cast<int>(num_threads/2)));
    } else {
        g_num_threads_per_camera = std::max(1, static_cast<int>(num_threads/4));
    }

    // Initialize serial communication - call directly without checking return value
    init_serial_comm("/dev/ttyS9");

    // Initialize Hikvision SDK
    if (!NET_DVR_Init()) {
        std::cerr << "Hikvision SDK initialization failed!" << std::endl;
        return -1;
    }
    NET_DVR_SetConnectTime(3000, 3);

    // Read camera configuration
    auto cameras = ReadCameraConfig(configFile);
    if (cameras.empty()) {
        std::cerr << "No valid camera configurations found!" << std::endl;
        NET_DVR_Cleanup();
        return -1;
    }

    std::cout << "Starting " << cameras.size() << " camera streams..." << std::endl;

    // Start camera threads
    std::vector<std::thread> threads;
    for (auto& camera : cameras) {
        threads.emplace_back(ProcessCameraStream, std::ref(camera));
    }

    // Main loop to update max count
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint16_t current_max = g_max_box_count.load();
        if (current_max > 0) {
            send_people_count(current_max);
            g_max_box_count.store(0);
        }
    }

    // Wait for threads to finish
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    // Cleanup database connections
    for (auto& [name, db] : db_pool) {
        sqlite3_close(db);
    }
    db_pool.clear();

    // Cleanup SDK
    NET_DVR_Cleanup();
    std::cout << "All camera streams stopped, program exiting" << std::endl;
    return 0;
}