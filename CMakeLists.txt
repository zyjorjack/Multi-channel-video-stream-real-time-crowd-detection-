# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# 设置项目名称
project(rk3588-demo VERSION 0.0.1 LANGUAGES C CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 在 project() 声明后添加
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

# 设置编译器标志
set(CMAKE_C_FLAGS "-g")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14")

# rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
# rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
# rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

find_package(X11 REQUIRED)

# 海康SDK配置
set(HIKVISION_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/hiklib)
set(HIKVISION_SDK_INCLUDE ${HIKVISION_SDK_PATH})
set(HIKVISION_SDK_LIBS 
    -L${HIKVISION_SDK_PATH} 
    -Wl,-rpath=${HIKVISION_SDK_PATH}
    -lhcnetsdk 
    -lPlayCtrl
    -lssl 
    -lcrypto
    -ldl
    X11
)

# 寻找OpenCV库
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# 用来搜索头文件的目录.h所在路径
# 包含路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${RGA_DIR}/include
    ${HIKVISION_SDK_INCLUDE}
)

# 链接目录
link_directories(
    ${HIKVISION_SDK_PATH}
    /usr/lib/aarch64-linux-gnu
)

# 构建预处理和后处理库
add_library(nn_process SHARED
            src/process/preprocess.cpp
            src/process/postprocess.cpp
)
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

# 构建自定义封装API库
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp) 
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)

# yolov8_lib
add_library(yolov8_lib SHARED src/task/yolov8_custom.cpp)
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process
)

# draw_lib
add_library(draw_lib SHARED src/draw/cv_draw.cpp)
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

# 测试yolov8 thread pool
add_executable(yolov8_thread_pool 
    src/yolov8_thread_pool.cpp
    src/task/yolov8_thread_pool.cpp
)
target_link_libraries(yolov8_thread_pool
    draw_lib
    yolov8_lib
    pthread
)

find_package(SQLite3 REQUIRED)

# 海康SDK多线程读流+YOLOv8推理
add_executable(yolov8_thread_pool_hik
    src/yolov8_thread_pool_hik.cpp
    src/task/yolov8_thread_pool.cpp
    src/task/comm.cpp  # 添加这一行
    src/task/mask_utils.cpp
)
target_link_libraries(yolov8_thread_pool_hik
    draw_lib
    yolov8_lib
    ${HIKVISION_SDK_LIBS}
    ${OpenCV_LIBS}
    pthread
    dl
    ${SQLite3_LIBRARIES}
)

# 安装规则
install(TARGETS 
    yolov8_thread_pool 
    yolov8_thread_pool_hik
    DESTINATION bin
)