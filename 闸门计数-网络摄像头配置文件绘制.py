import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QFileDialog, QMessageBox,
                             QLineEdit, QGroupBox, QSizePolicy, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon


class PolygonDrawer(QWidget):
    def __init__(self):
        super().__init__()
        self.current_polygon = []  # 当前正在绘制的多边形（显示坐标）
        self.polygons = []         # 已完成的多边形（原始图像坐标）
        self.drawing = False
        self.frame = None
        self.mask_alpha = 0.2
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 添加用于坐标转换的变量
        self.orig_width = 0
        self.orig_height = 0
        self.display_width = 0
        self.display_height = 0
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_frame(self, frame):
        self.frame = frame.copy()
        self.orig_height, self.orig_width = self.frame.shape[:2]
        self.update_display_parameters()
        self.update()

    def update_display_parameters(self):
        """更新显示参数（在设置帧或窗口大小变化时调用）"""
        if self.frame is None:
            return
            
        widget_width = self.width()
        widget_height = self.height()
        
        # 计算保持宽高比的缩放比例
        scale_w = widget_width / self.orig_width
        scale_h = widget_height / self.orig_height
        self.scale_factor = min(scale_w, scale_h)
        
        # 计算显示尺寸
        self.display_width = int(self.orig_width * self.scale_factor)
        self.display_height = int(self.orig_height * self.scale_factor)
        
        # 计算偏移量（居中显示）
        self.offset_x = (widget_width - self.display_width) // 2
        self.offset_y = (widget_height - self.display_height) // 2

    def resizeEvent(self, event):
        """处理窗口大小变化"""
        super().resizeEvent(event)
        self.update_display_parameters()
        self.update()

    def _to_display_coords(self, point):
        """将原始图像坐标转换为显示坐标"""
        x = int(point.x() * self.scale_factor) + self.offset_x
        y = int(point.y() * self.scale_factor) + self.offset_y
        return QPoint(x, y)

    def _to_orig_coords(self, point):
        """将显示坐标转换为原始图像坐标"""
        x = int((point.x() - self.offset_x) / self.scale_factor)
        y = int((point.y() - self.offset_y) / self.scale_factor)
        return QPoint(max(0, min(x, self.orig_width)), 
                     max(0, min(y, self.orig_height)))

    def _check_boundary(self, pos):
        """确保点在显示区域内"""
        x = max(self.offset_x, min(pos.x(), self.offset_x + self.display_width))
        y = max(self.offset_y, min(pos.y(), self.offset_y + self.display_height))
        return QPoint(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            checked_pos = self._check_boundary(event.pos())
            self.current_polygon = [checked_pos]
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() & Qt.LeftButton:
            checked_pos = self._check_boundary(event.pos())
            if self.current_polygon:
                # 避免添加太近的点
                last_point = self.current_polygon[-1]
                if (abs(last_point.x() - checked_pos.x()) > 5 or 
                    abs(last_point.y() - checked_pos.y()) > 5):
                    self.current_polygon.append(checked_pos)
                    self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if len(self.current_polygon) > 2:  # 至少3个点才能形成多边形
                checked_pos = self._check_boundary(event.pos())
                if self.current_polygon:
                    last_point = self.current_polygon[-1]
                    if (abs(last_point.x() - checked_pos.x()) > 5 or 
                        abs(last_point.y() - checked_pos.y()) > 5):
                        self.current_polygon.append(checked_pos)
                
                # 转换为原始坐标并存储
                orig_polygon = [self._to_orig_coords(p) for p in self.current_polygon]
                self.polygons.append([(p.x(), p.y()) for p in orig_polygon])
            
            self.current_polygon = []
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.frame is not None:
            # 创建缩放后的图像
            scaled_frame = cv2.resize(
                self.frame, 
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_AREA
            )
            
            # 显示视频帧
            height, width, channel = scaled_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(scaled_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            painter.drawPixmap(self.offset_x, self.offset_y, QPixmap.fromImage(q_img))
            
            # 绘制半透明掩膜
            if self.polygons:
                # 创建透明层
                overlay = QPixmap(self.width(), self.height())
                overlay.fill(Qt.transparent)
                overlay_painter = QPainter(overlay)
                overlay_painter.setRenderHint(QPainter.Antialiasing)
                
                # 设置半透明颜色
                color = QColor(255, 0, 0, int(255 * self.mask_alpha))
                overlay_painter.setBrush(color)
                overlay_painter.setPen(Qt.NoPen)
                
                # 绘制所有多边形
                for polygon in self.polygons:
                    display_points = [self._to_display_coords(QPoint(x, y)) for x, y in polygon]
                    qpoly = QPolygon(display_points)
                    overlay_painter.drawPolygon(qpoly)
                
                overlay_painter.end()
                painter.drawPixmap(0, 0, overlay)
            
            # 绘制多边形轮廓
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            
            # 绘制已完成的polygons
            for polygon in self.polygons:
                display_points = [self._to_display_coords(QPoint(x, y)) for x, y in polygon]
                qpoly = QPolygon(display_points)
                painter.drawPolygon(qpoly)
            
            # 绘制当前正在绘制的polygon
            if len(self.current_polygon) > 1:
                # 使用QPolygon绘制折线
                qpoly = QPolygon(self.current_polygon)
                painter.drawPolyline(qpoly)
                
                # 绘制起点和终点之间的连线（形成闭合）
                if len(self.current_polygon) > 2:
                    painter.drawLine(self.current_polygon[-1], self.current_polygon[0])

    def clear_polygons(self):
        self.polygons = []
        self.update()

    def delete_last_polygon(self):
        if self.polygons:
            self.polygons.pop()
            self.update()

    def get_polygons(self):
        """获取原始图像坐标的多边形"""
        return self.polygons
    
    def set_polygons(self, polygons, orig_width, orig_height):
        """设置多边形（使用原始图像坐标）"""
        if orig_width != self.orig_width or orig_height != self.orig_height:
            # 如果图像尺寸变化，需要缩放多边形
            scale_x = self.orig_width / orig_width
            scale_y = self.orig_height / orig_height
            self.polygons = []
            for polygon in polygons:
                scaled_poly = [(int(x * scale_x), int(y * scale_y)) for (x, y) in polygon]
                self.polygons.append(scaled_poly)
        else:
            # 图像尺寸相同，直接使用
            self.polygons = polygons
        self.update()


class CameraMaskEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头掩膜编辑工具")
        self.setGeometry(100, 100, 1200, 800)

        self.current_config_path = None
        self.camera_list = []
        self.current_camera_index = -1
        self.rtsp_url = ""
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧控制面板 (固定宽度)
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout()

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.load_btn = QPushButton("打开配置文件")
        self.load_btn.clicked.connect(self.load_config_file)
        file_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("保存配置文件")
        self.save_btn.clicked.connect(self.save_config_file)
        file_layout.addWidget(self.save_btn)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # 摄像头列表
        camera_list_group = QGroupBox("摄像头列表")
        camera_list_layout = QVBoxLayout()
        self.camera_list_widget = QListWidget()
        self.camera_list_widget.itemClicked.connect(self.select_camera)
        camera_list_layout.addWidget(self.camera_list_widget)
        camera_list_group.setLayout(camera_list_layout)
        control_layout.addWidget(camera_list_group)

        # 摄像头信息
        self.camera_info_group = QGroupBox("摄像头信息")
        camera_info_layout = QVBoxLayout()

        self.ip_label = QLabel("IP地址:")
        self.ip_edit = QLineEdit()
        self.ip_edit.setReadOnly(True)
        camera_info_layout.addWidget(self.ip_label)
        camera_info_layout.addWidget(self.ip_edit)

        self.user_label = QLabel("用户名:")
        self.user_edit = QLineEdit()
        self.user_edit.setReadOnly(True)
        camera_info_layout.addWidget(self.user_label)
        camera_info_layout.addWidget(self.user_edit)

        self.pass_label = QLabel("密码:")
        self.pass_edit = QLineEdit()
        self.pass_edit.setReadOnly(True)
        self.pass_edit.setEchoMode(QLineEdit.Password)
        camera_info_layout.addWidget(self.pass_label)
        camera_info_layout.addWidget(self.pass_edit)

        self.channel_label = QLabel("通道号:")
        self.channel_edit = QLineEdit()
        self.channel_edit.setReadOnly(True)
        camera_info_layout.addWidget(self.channel_label)
        camera_info_layout.addWidget(self.channel_edit)

        self.resolution_label = QLabel("分辨率:")
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920*1080", "1280*720", "704*576", "352*288"])
        camera_info_layout.addWidget(self.resolution_label)
        camera_info_layout.addWidget(self.resolution_combo)

        self.camera_info_group.setLayout(camera_info_layout)
        control_layout.addWidget(self.camera_info_group)

        # 多边形操作
        poly_group = QGroupBox("多边形掩膜操作")
        poly_layout = QVBoxLayout()

        self.connect_btn = QPushButton("连接摄像头")
        self.connect_btn.clicked.connect(self.connect_current_camera)
        poly_layout.addWidget(self.connect_btn)

        self.load_image_btn = QPushButton("加载本地图片")
        self.load_image_btn.clicked.connect(self.load_local_image)
        poly_layout.addWidget(self.load_image_btn)

        btn_layout = QHBoxLayout()
        self.add_poly_btn = QPushButton("新增多边形")
        self.add_poly_btn.clicked.connect(self.enable_drawing)
        btn_layout.addWidget(self.add_poly_btn)

        self.del_poly_btn = QPushButton("删除最后")
        self.del_poly_btn.clicked.connect(self.delete_last_polygon)
        btn_layout.addWidget(self.del_poly_btn)
        poly_layout.addLayout(btn_layout)

        self.clear_poly_btn = QPushButton("清空所有")
        self.clear_poly_btn.clicked.connect(self.clear_polygons)
        poly_layout.addWidget(self.clear_poly_btn)

        self.save_camera_btn = QPushButton("保存当前配置")
        self.save_camera_btn.clicked.connect(self.save_current_camera)
        poly_layout.addWidget(self.save_camera_btn)

        poly_group.setLayout(poly_layout)
        control_layout.addWidget(poly_group)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # 右侧视频显示区域
        self.video_display = PolygonDrawer()
        main_layout.addWidget(self.video_display, stretch=1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 初始状态
        self.update_ui_state()

    def update_ui_state(self):
        has_camera = self.current_camera_index >= 0
        self.connect_btn.setEnabled(has_camera)
        self.load_image_btn.setEnabled(has_camera)
        self.add_poly_btn.setEnabled(has_camera)
        self.del_poly_btn.setEnabled(has_camera)
        self.clear_poly_btn.setEnabled(has_camera)
        self.save_camera_btn.setEnabled(has_camera)

    def load_config_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "",
                                                   "Text Files (*.txt);;All Files (*)",
                                                   options=options)
        if file_path:
            self.current_config_path = file_path
            self.camera_list = self.parse_config_file(file_path)
            self.update_camera_list()

    def parse_config_file(self, file_path):
        cameras = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # 解析基础信息
                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    ip = parts[0]
                    username = parts[1]
                    password = parts[2]
                    channel = parts[3]

                    # 默认分辨率
                    resolution = "1920*1080"
                    polygons = []

                    # 检查是否有分辨率
                    if len(parts) > 4:
                        # 检查第五部分是否是分辨率
                        if '*' in parts[4]:
                            resolution = parts[4]
                            polygon_parts = parts[5:]
                        else:
                            polygon_parts = parts[4:]

                        # 解析多边形
                        polygon_str = ' '.join(polygon_parts)
                        polygons = self.parse_polygons(polygon_str)

                    cameras.append({
                        'ip': ip,
                        'username': username,
                        'password': password,
                        'channel': channel,
                        'resolution': resolution,
                        'polygons': polygons
                    })
        except Exception as e:
            QMessageBox.warning(self, "错误", f"解析配置文件失败: {str(e)}")

        return cameras

    def parse_polygons(self, polygon_str):
        polygons = []
        start = 0
        while True:
            start_bracket = polygon_str.find('[', start)
            if start_bracket == -1:
                break
            end_bracket = polygon_str.find(']', start_bracket)
            if end_bracket == -1:
                break

            points_str = polygon_str[start_bracket + 1:end_bracket]
            points = []
            for point_str in points_str.split():
                x, y = map(int, point_str.split(','))
                points.append((x, y))

            if len(points) >= 3:  # 至少3个点才能形成多边形
                polygons.append(points)

            start = end_bracket + 1

        return polygons

    def update_camera_list(self):
        self.camera_list_widget.clear()
        for cam in self.camera_list:
            self.camera_list_widget.addItem(f"{cam['ip']} (Ch{cam['channel']})")

    def select_camera(self, item):
        index = self.camera_list_widget.row(item)
        if 0 <= index < len(self.camera_list):
            self.current_camera_index = index
            cam = self.camera_list[index]

            self.ip_edit.setText(cam['ip'])
            self.user_edit.setText(cam['username'])
            self.pass_edit.setText(cam['password'])
            self.channel_edit.setText(cam['channel'])
            self.resolution_combo.setCurrentText(cam['resolution'])

            # 设置多边形
            self.video_display.clear_polygons()
            
            # 如果已经加载了图像，设置多边形
            if self.video_display.frame is not None:
                # 从配置中获取原始分辨率
                res_text = cam['resolution']
                orig_width, orig_height = map(int, res_text.split('*'))
                
                # 设置多边形（自动处理尺寸变化）
                self.video_display.set_polygons(
                    cam['polygons'], 
                    orig_width, 
                    orig_height
                )
            self.update_ui_state()

    def connect_current_camera(self):
        if self.current_camera_index < 0:
            return

        cam = self.camera_list[self.current_camera_index]

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.timer.stop()

        # 构建RTSP URL (主流)
        channel = int(cam['channel'])
        rtsp_channel = (channel - 1) * 100 + 1  # 101, 201, 301等
        self.rtsp_url = f"rtsp://{cam['username']}:{cam['password']}@{cam['ip']}:554//Streaming/Channels/{rtsp_channel}"

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")

            self.timer.start(30)  # 30ms更新一帧
        except Exception as e:
            QMessageBox.warning(self, "错误", f"连接摄像头失败: {str(e)}")
            self.cap = None

    def load_local_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "",
                                                   "Images (*.png *.jpg *.bmp);;All Files (*)",
                                                   options=options)
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                self.video_display.set_frame(frame)
                
                # 如果选择了摄像头，加载其多边形
                if 0 <= self.current_camera_index < len(self.camera_list):
                    cam = self.camera_list[self.current_camera_index]
                    res_text = cam['resolution']
                    orig_width, orig_height = map(int, res_text.split('*'))
                    
                    self.video_display.set_polygons(
                        cam['polygons'], 
                        orig_width, 
                        orig_height
                    )

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 获取当前分辨率设置
                res_text = self.resolution_combo.currentText()
                width, height = map(int, res_text.split('*'))
                frame = cv2.resize(frame, (width, height))
                self.video_display.set_frame(frame)
                
                # 如果选择了摄像头，加载其多边形
                if 0 <= self.current_camera_index < len(self.camera_list):
                    cam = self.camera_list[self.current_camera_index]
                    res_text = cam['resolution']
                    orig_width, orig_height = map(int, res_text.split('*'))
                    
                    self.video_display.set_polygons(
                        cam['polygons'], 
                        orig_width, 
                        orig_height
                    )

    def enable_drawing(self):
        if self.video_display.frame is None:
            QMessageBox.warning(self, "警告", "请先连接摄像头或加载图片")
            return

    def delete_last_polygon(self):
        self.video_display.delete_last_polygon()

    def clear_polygons(self):
        self.video_display.clear_polygons()

    def save_current_camera(self):
        if self.current_camera_index < 0:
            return

        # 更新当前摄像头信息
        cam = self.camera_list[self.current_camera_index]
        cam['resolution'] = self.resolution_combo.currentText()
        cam['polygons'] = self.video_display.get_polygons()

        QMessageBox.information(self, "成功", "当前摄像头配置已更新")

    def save_config_file(self):
        if not self.camera_list:
            QMessageBox.warning(self, "警告", "没有可保存的摄像头配置")
            return

        options = QFileDialog.Options()
        default_path = self.current_config_path if self.current_config_path else ""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存配置文件", default_path,
                                                   "Text Files (*.txt);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    for cam in self.camera_list:
                        line_parts = [
                            cam['ip'],
                            cam['username'],
                            cam['password'],
                            cam['channel'],
                            cam['resolution']
                        ]

                        # 添加多边形
                        for polygon in cam['polygons']:
                            poly_str = '[' + ' '.join(f"{x},{y}" for x, y in polygon) + ']'
                            line_parts.append(poly_str)

                        f.write(' '.join(line_parts) + '\n')

                QMessageBox.information(self, "成功", f"配置文件已保存到: {file_path}")
                self.current_config_path = file_path
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存配置文件失败: {str(e)}")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraMaskEditor()
    window.show()
    sys.exit(app.exec_())