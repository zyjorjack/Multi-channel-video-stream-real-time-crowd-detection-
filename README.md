编译：
rm -rf build  # 如果项目名称和内容发生变化，删除旧的 build 目录（包括 CMakeCache.txt）
cmake -S . -B build  # 重新生成
cmake --build build/

# 修改458串口号：
在src/yolov8_thread_pool_hik.cpp中 init_serial_comm("/dev/tty0");
# 权限，串口
sudo chmod 666 /dev/tty0

# 基于海康SDK的运行命令（程序地址 模型地址 海康摄像头配置文件 线程数量）
./build/yolov8_thread_pool_hik ./weights/yolov8s.int.rknn cameras_config.txt 30
./build/yolov8_thread_pool_hik ./weights/Gate_people_counting_8n_int.rknn cameras_config.txt 20
./build/yolov8_thread_pool_hik ./weights/Gate_people_countingv32_8n_int.rknn cameras_config.txt 20

# 查看数据库内容
# 查看检测结果表的所有数据：
sqlite3 detection_results.db "SELECT * FROM detection_results ORDER BY id DESC;"
# 查看最新的5条记录：
sqlite3 detection_results.db "SELECT * FROM detection_results ORDER BY id DESC LIMIT 5;"
# 查看表结构：
sqlite3 detection_results.db ".schema detection_results"

✅ 多路摄像头支持：可同时接入多个海康摄像头，支持自定义配置
✅ YOLOv8 实时检测：基于 Ultralytics YOLOv8 的高精度人体检测
✅ 线程池并行处理：每路视频流独立线程池，最大化利用多核CPU
✅ 区域排除功能：支持设置屏蔽区域，避免误检
✅ 数据持久化：检测结果自动存入 SQLite 数据库和 485 协议传输，支持历史查询
✅ 实时显示与监控：使用 OpenCV 实时显示检测画面，支持画面标注与计数展示

📦 适用场景
商场/超市人流量统计
地铁站/火车站实时监控
场馆/展览会人流管理
智慧园区安防监控
其他需要多路视频AI分析的场景
