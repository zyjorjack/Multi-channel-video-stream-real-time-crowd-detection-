#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <fcntl.h>
#include <termios.h>
#include <stdlib.h>
#include <sqlite3.h>
#include <time.h>
#include <vector>
#include <string>
#include <chrono>
#include <sys/stat.h>

#include <sqlite3.h>
#include <string>

sqlite3* GetDatabaseConnection(const std::string& db_name);

#define FRAME_MAX 128
#define SLAVE_ADDR 0x01
#define MASTER_ADDR 0x00

// 全局变量
static int fd = -1;
static uint16_t current_people = 0;
static sqlite3* send_db = nullptr;
static std::string g_max_device_id;
static std::string g_max_timestamp;
static pthread_mutex_t db_mutex = PTHREAD_MUTEX_INITIALIZER;

// 获取当前时间戳字符串
static std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    
    char buf[64];
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &bt);
    
    char result[64];
    snprintf(result, sizeof(result), "%s%03ld", buf, ms.count());
    
    return result;
}

// 初始化发送数据库（增强版）
// static bool init_send_db() {
//     std::string db_path = "./detection_results_send.db";
//     if (access(db_path.c_str(), F_OK) != 0) {
//         FILE* f = fopen(db_path.c_str(), "w");
//         if (f) fclose(f);
//         else {
//             perror("Failed to create database file");
//             pthread_mutex_unlock(&db_mutex);
//             return false;
//         }
//     }


static bool init_send_db() {
    pthread_mutex_lock(&db_mutex);
    
    // 使用新接口获取连接
    send_db = GetDatabaseConnection("detection_results_send.db");
    if (!send_db) {
        pthread_mutex_unlock(&db_mutex);
        return false;
    }

    // 检查表是否存在
    const char* check_sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='detection_results';";
    sqlite3_stmt* stmt;
    bool table_exists = false;
    
    if (sqlite3_prepare_v2(send_db, check_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            table_exists = true;
        }
        sqlite3_finalize(stmt);
    }

    // 创建表（如果不存在）
    if (!table_exists) {
        const char* create_sql = "CREATE TABLE detection_results ("
                               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                               "device TEXT NOT NULL,"
                               "timestamp TEXT NOT NULL,"
                               "box_count INTEGER NOT NULL);";
        
        char* errMsg = nullptr;
        if (sqlite3_exec(send_db, create_sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
            fprintf(stderr, "Failed to create table: %s\n", errMsg);
            sqlite3_free(errMsg);
            pthread_mutex_unlock(&db_mutex);
            return false;
        }
    }

    // 优化设置
    sqlite3_exec(send_db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(send_db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    
    pthread_mutex_unlock(&db_mutex);
    return true;
}

// 计算LRC校验码
static uint8_t calc_lrc(const uint8_t* data, size_t len) {
    uint8_t lrc = 0;
    for (size_t i = 0; i < len; ++i) {
        lrc += data[i];
    }
    return (uint8_t)(-((int8_t)lrc));
}

// 检查LRC校验码
static bool check_lrc(const uint8_t* data, size_t len) {
    if (len < 4) return false;
    uint8_t calc = calc_lrc(&data[1], len - 4);
    uint8_t recv = data[len - 3];
    if (calc != recv) {
        printf("[LRC ERROR] Calc: %02X, Recv: %02X\n", calc, recv);
    }
    return calc == recv;
}

// 构建响应帧
static int build_response_frame(uint8_t from_addr, uint8_t to_addr, uint8_t func_code, 
                              const uint8_t* data, uint16_t data_len, uint8_t* out_buf) {
    int idx = 0;
    out_buf[idx++] = 0x3A;       // 开始符
    out_buf[idx++] = from_addr;  // 源地址
    out_buf[idx++] = to_addr;    // 目的地址
    out_buf[idx++] = func_code;  // 功能码
    
    // 数据位数 (16位)
    out_buf[idx++] = (data_len >> 8) & 0xFF;
    out_buf[idx++] = data_len & 0xFF;
    
    // 数据
    if (data_len > 0 && data != nullptr) {
        memcpy(&out_buf[idx], data, data_len);
        idx += data_len;
    }
    
    // LRC校验码
    out_buf[idx++] = calc_lrc(&out_buf[1], idx - 1);
    
    // 结束符
    out_buf[idx++] = 0x0D;
    out_buf[idx++] = 0x0A;
    
    return idx;
}

// 查询历史记录
static std::vector<uint8_t> query_history_records(const std::string& start_time, const std::string& end_time) {
    std::vector<uint8_t> result;
    pthread_mutex_lock(&db_mutex);
    
    if (!send_db) {
        pthread_mutex_unlock(&db_mutex);
        return result;
    }

    const char* sql = R"(
        SELECT SUM(box_count) FROM detection_results 
        WHERE timestamp BETWEEN ? AND ? AND box_count > 0;
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(send_db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, start_time.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, end_time.c_str(), -1, SQLITE_TRANSIENT);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int total = sqlite3_column_int(stmt, 0);
            // printf("[DB] Query %s~%s => %d people\n", 
            //       start_time.c_str(), end_time.c_str(), total);
            result.push_back((total >> 8) & 0xFF);
            result.push_back(total & 0xFF);
        } else {
            // printf("[DB] No data found for %s~%s\n", 
            //       start_time.c_str(), end_time.c_str());
        }
        sqlite3_finalize(stmt);
    } else {
        printf("[DB ERROR] %s\n", sqlite3_errmsg(send_db));
    }
    
    pthread_mutex_unlock(&db_mutex);
    return result.empty() ? std::vector<uint8_t>{0x00, 0x00} : result;
}

// 保存发送记录到数据库（增强版）
static void save_to_send_db(const std::string& device, const std::string& timestamp, uint16_t count) {
    pthread_mutex_lock(&db_mutex);
    
    // 确保数据库已初始化
    if (!send_db && !init_send_db()) {
        pthread_mutex_unlock(&db_mutex);
        return;
    }

    const char* sql = "INSERT INTO detection_results (device, timestamp, box_count) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(send_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        printf("[DB ERROR] Prepare statement failed: %s\n", sqlite3_errmsg(send_db));
        pthread_mutex_unlock(&db_mutex);
        return;
    }
    
    sqlite3_bind_text(stmt, 1, device.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, count);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        printf("[DB ERROR] Insert failed: %s\n", sqlite3_errmsg(send_db));
    } else {
        printf("[DB] Successfully saved the record: %s @ %s -> %d\n", 
              device.c_str(), timestamp.c_str(), count);
    }
    
    sqlite3_finalize(stmt);
    pthread_mutex_unlock(&db_mutex);
}

// 接收线程（完整版）
static void* recv_thread(void* arg) {
    uint8_t buf[FRAME_MAX];
    
    // 确保数据库初始化成功
    if (!init_send_db()) {
        fprintf(stderr, "Unable to initialize the sending database, communication functions will be limited\n");
    }

    while (1) {
        int len = read(fd, buf, sizeof(buf));
        if (len <= 0) {
            usleep(10000);
            continue;
        }

        // 打印原始数据
        printf("[RX] ");
        for (int i = 0; i < len; i++) printf("%02X ", buf[i]);
        printf("\n");

        // 基本帧检查
        if (buf[0] != 0x3A || len < 9) {
            printf("[ERROR] Invalid frame header\n");
            continue;
        }

        // LRC校验
        if (!check_lrc(buf, len)) {
            printf("[ERROR] LRC check failed\n");
            continue;
        }

        // 解析帧
        uint8_t to_addr = buf[1];
        uint8_t from_addr = buf[2];
        uint8_t func_code = buf[3];
        uint16_t data_len = (buf[4] << 8) | buf[5];

        // 只处理发给本机的帧
        if (to_addr != SLAVE_ADDR) {
            printf("[WARN] Ignore frame for addr %02X\n", to_addr);
            continue;
        }

        // 处理实时查询 (0x01)
        if (func_code == 0x01 && data_len == 0) {
            // printf("[CMD] Current people query\n");
            uint8_t resp[16];
            uint8_t data[2] = {
                static_cast<uint8_t>((current_people >> 8) & 0xFF),
                static_cast<uint8_t>(current_people & 0xFF)
            };
            int resp_len = build_response_frame(SLAVE_ADDR, MASTER_ADDR, 0x02, data, 2, resp);
            
            // 保存发送记录（无论current_people是否为0都保存）
            std::string device = g_max_device_id.empty() ? "unknown_device" : g_max_device_id;
            std::string timestamp = g_max_timestamp.empty() ? get_current_timestamp() : g_max_timestamp;
            save_to_send_db(device, timestamp, current_people);
            
            write(fd, resp, resp_len);
            printf("[TX] ");
            for (int i = 0; i < resp_len; i++) printf("%02X ", resp[i]);
            printf("\n");
        }
        // 处理历史查询 (0x03)
        else if (func_code == 0x03 && data_len == 28) {
            std::string start_time((const char*)&buf[6], 14);
            std::string end_time((const char*)&buf[20], 14);
            printf("[CMD] History query: %s to %s\n", start_time.c_str(), end_time.c_str());

            auto history_data = query_history_records(start_time, end_time);
            uint8_t resp[32];
            int resp_len = build_response_frame(SLAVE_ADDR, MASTER_ADDR, 0x04, 
                                              history_data.data(), history_data.size(), resp);
            write(fd, resp, resp_len);
            printf("[TX] ");
            for (int i = 0; i < resp_len; i++) printf("%02X ", resp[i]);
            printf("\n");
        }
        else {
            printf("[ERROR] Unsupported command: func=%02X len=%d\n", func_code, data_len);
        }
    }
    return NULL;
}

// 串口设置
static int set_serial_opt(int fd, int baudrate, int databits, char parity, int stopbits) {
    struct termios options;
    if (tcgetattr(fd, &options) != 0) {
        perror("tcgetattr");
        return -1;
    }

    cfsetispeed(&options, baudrate);
    cfsetospeed(&options, baudrate);

    options.c_cflag &= ~CSIZE;
    switch (databits) {
        case 7: options.c_cflag |= CS7; break;
        case 8: options.c_cflag |= CS8; break;
        default: options.c_cflag |= CS8; break;
    }

    switch (parity) {
        case 'n': case 'N':
            options.c_cflag &= ~PARENB;
            break;
        case 'o': case 'O':
            options.c_cflag |= PARENB | PARODD;
            break;
        case 'e': case 'E':
            options.c_cflag |= PARENB;
            options.c_cflag &= ~PARODD;
            break;
        default:
            options.c_cflag &= ~PARENB;
            break;
    }

    options.c_cflag &= ~CSTOPB;
    if (stopbits == 2) options.c_cflag |= CSTOPB;

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_iflag &= ~(INPCK | ISTRIP | IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 10;

    tcflush(fd, TCIFLUSH);
    if (tcsetattr(fd, TCSANOW, &options) != 0) {
        perror("tcsetattr");
        return -1;
    }
    return 0;
}

// 初始化串口通信
void init_serial_comm(const char* device) {
    // 先初始化数据库
    if (!init_send_db()) {
        fprintf(stderr, "Warning: Database initialization failed, some functions may be limited\n");
    }

    fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
        perror("open serial failed");
        exit(1);
    }

    if (set_serial_opt(fd, B115200, 8, 'N', 1) < 0) {
        close(fd);
        exit(1);
    }

    pthread_t tid;
    pthread_create(&tid, NULL, recv_thread, NULL);
    pthread_detach(tid);
}

// 更新要发送的人数
void send_people_count(uint16_t people) {
    current_people = people;
    // printf("[PEOPLE] Set current people: %d\n", people);
}

// 更新最大检测信息
void update_max_info(const std::string& device_id, const std::string& timestamp) {
    g_max_device_id = device_id;
    g_max_timestamp = timestamp;
    // printf("[UPDATE] Max info: %s @ %s\n", device_id.c_str(), timestamp.c_str());
}