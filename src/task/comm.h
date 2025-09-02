#ifndef COMM_H
#define COMM_H

#include <stdint.h>
#pragma once
#include <sqlite3.h>
#include <string>

sqlite3* GetDatabaseConnection(const std::string& db_name);
void init_serial_comm(const char* device);  // 初始化串口通信线程
void send_people_count(uint16_t people);    // 设置要返回的人数
void update_max_info(const std::string& device_id, const std::string& timestamp);
#endif
