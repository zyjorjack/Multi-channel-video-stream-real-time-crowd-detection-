import cv2

# 读取图像
img = cv2.imread("mask_192.168.1.103_ch1.png")
if img is None:
    print("Error: Image not found")
    exit()

h, w = img.shape[:2]

# 设置显示缩放比例 (例如0.5表示缩小一半)
scale = 0.5

# 创建缩放后的图像用于显示
display_img = cv2.resize(img, (int(w * scale), int(h * scale)))


# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 计算原始图像坐标
        original_x = int(x / scale)
        original_y = int(y / scale)

        # 在图像上显示坐标
        temp_img = display_img.copy()
        cv2.putText(temp_img, f"({original_x}, {original_y})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('Scaled Image', temp_img)

        print(f"Original coordinates: ({original_x}, {original_y})")


# 创建窗口并设置回调
cv2.namedWindow('Scaled Image')
cv2.setMouseCallback('Scaled Image', mouse_callback)

# 显示缩放后的图像
cv2.imshow('Scaled Image', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()