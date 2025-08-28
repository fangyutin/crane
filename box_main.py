import cv2
import time
import os
import subprocess
import yaml
import shutil
import serial
import threading
from glob import glob

# 配置路径
YOLOV5_DIR = "/home/pi/yolo/yolov5-master"
OUTPUT_DIR = "/home/pi/yolo/outcome"
TEMP_DIR = os.path.join(YOLOV5_DIR, "temp_images")
os.makedirs(TEMP_DIR, exist_ok=True)

# 全局变量用于存储当前检测结果和串口状态
current_result = ""
serial_active = True


def apply_digital_zoom(frame, zoom_factor=1.3, center_x=0.5, center_y=0.65):
    """应用数码变焦"""
    h, w = frame.shape[:2]
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)

    # 计算中心点坐标
    cx = int(w * center_x)
    cy = int(h * center_y)

    # 计算裁剪区域
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # 裁剪并缩放
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def capture_image(cap, warmup_frames=10):
    """拍摄单张图片并应用数码变焦"""
    # 增加曝光预热帧
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.1)

    # 正式拍摄
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("摄像头读取失败")

    zoom_frame = apply_digital_zoom(frame)
    return zoom_frame


def load_class_names():
    """加载类别名称"""
    with open(os.path.join(YOLOV5_DIR, "data", "coco.yaml")) as f:
        data = yaml.safe_load(f)
        return data['names']


def get_detected_labels(label_paths, names, img_size):
    """获取检测结果标签，按六个固定区域排序，返回六位字符串（每个区域取最左对象）"""
    img_w, img_h = img_size
    results = []

    # 定义六个区域的归一化边界 (x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio)
    region_bounds = [
        # 区域 a
        (0.0, 0.219, 0.2, 0.5),
        # 区域 b
        (0.344, 0.609, 0.208, 0.5),
        # 区域 c
        (0.703, 1, 0.208, 0.5),
        # 区域 d
        (0.0, 0.219, 0.5, 1),
        # 区域 e
        (0.344, 0.609, 0.5, 1),
        # 区域 f
        (0.703, 1, 0.5, 1)
    ]

    for path in label_paths:
        if not os.path.exists(path):
            # 如果文件不存在，返回六个'x'
            results.append(['x'] * 6)
            continue

        objects = []  # 存储检测到的对象信息

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # 解析数据
                class_id = int(parts[0])
                # 确保类别ID在有效范围内
                if class_id < 0 or class_id >= len(names):
                    continue

                # 获取类别名称并确保为字符串
                class_name = str(names[class_id])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 转换为像素坐标
                x_center_px = x_center * img_w
                y_center_px = y_center * img_h
                width_px = width * img_w
                height_px = height * img_h

                # 计算左上角坐标和中心点（用于区域判断）
                x_min = x_center_px - width_px / 2
                y_min = y_center_px - height_px / 2
                obj_center_x = x_center_px  # 使用中心点坐标进行区域判断
                obj_center_y = y_center_px

                # 添加对象信息（包含中心点坐标）
                objects.append({
                    'name': class_name,
                    'x_min': x_min,
                    'center_x': obj_center_x,
                    'center_y': obj_center_y
                })

        region_results = []  # 存储六个区域的结果

        # 处理每个区域
        for bounds in region_bounds:
            x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio = bounds
            # 计算当前区域的像素边界
            x_min_region = x_min_ratio * img_w
            x_max_region = x_max_ratio * img_w
            y_min_region = y_min_ratio * img_h
            y_max_region = y_max_ratio * img_h

            # 筛选在当前区域内的对象（使用中心点判断）
            in_region_objs = [
                obj for obj in objects
                if (x_min_region <= obj['center_x'] <= x_max_region) and
                   (y_min_region <= obj['center_y'] <= y_max_region)
            ]

            if not in_region_objs:
                region_results.append('x')  # 无对象时用'x'
            else:
                # 选择区域内最左侧对象（x_min最小）
                leftmost_obj = min(in_region_objs, key=lambda x: x['x_min'])
                region_results.append(leftmost_obj['name'])

        results.append(region_results)

    processed_results = []
    for res in results:
        processed = get_norepeat_string_box(res)  # 处理每个检测结果
        processed_results.append(processed)

    return processed_results



def get_norepeat_string_box(char_list):
    """处理单个检测结果（6个字符的列表），确保无重复数字"""
    # 将输入复制避免修改原始数据
    chars = char_list[:]
    all_digits = set('123456')

    # 步骤1: 消除重复数字 (忽略'x')
    seen = set()
    missing_set = all_digits - set(ch for ch in chars if ch != 'x')

    for i in range(len(chars)):
        if chars[i] == 'x':
            continue
        if chars[i] in seen:
            if missing_set:
                new_digit = missing_set.pop()
                chars[i] = new_digit
                seen.add(new_digit)
        else:
            seen.add(chars[i])

    # 步骤2: 替换'x'为缺失数字
    current_digits = set(ch for ch in chars if ch != 'x')
    missing_for_x = all_digits - current_digits

    for i in range(len(chars)):
        if chars[i] == 'x' and missing_for_x:
            new_digit = missing_for_x.pop()
            chars[i] = new_digit

    return ''.join(chars)


def detect_single_image(cap, names, img_size):
    """拍摄并检测单张图片，返回识别结果的字符串"""
    # 清空临时目录 - 确保每次只处理最新照片
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 清空YOLOv5输出目录 - 新增：确保只保留最新结果
    detection_output = os.path.join(OUTPUT_DIR, "detections")
    if os.path.exists(detection_output):
        shutil.rmtree(detection_output, ignore_errors=True)

    # 拍摄单张照片
    frame = capture_image(cap)
    img_path = os.path.join(TEMP_DIR, "temp.jpg")
    cv2.imwrite(img_path, frame)

    # 运行YOLOv5检测
    subprocess.run([
        '/home/pi/PycharmProjects/box/.venv/bin/python',
        os.path.join(YOLOV5_DIR, "detect.py"),
        '--source', TEMP_DIR,
        '--project', OUTPUT_DIR,
        '--name', 'detections',
        '--exist-ok',
        '--save-txt',
        '--save-conf'
    ], cwd=YOLOV5_DIR, check=True)

    # 获取标签路径 - 注意YOLOv5生成的标签文件名
    # 使用基本文件名而不是带扩展名的完整文件名
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(OUTPUT_DIR, "detections", "labels", f"{base_name}.txt")

    # 处理检测结果
    detected_labels = get_detected_labels([label_path], names, img_size)

    # 返回当前图片的识别结果字符串
    if detected_labels:
        return detected_labels[0]  # 返回第一个(唯一)处理后的字符串
    return ""


def serial_sender(port='/dev/serial0', baudrate=115200):
    """持续通过串口发送当前检测结果"""
    global current_result, serial_active

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print("串口已打开，开始发送数据...")

        while serial_active:
            if current_result:
                ser.write(current_result.encode() + b'\n')
                print(f"已发送: {current_result}")
            time.sleep(1)  # 每秒发送一次

    except Exception as e:
        print(f"串口通信错误: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("串口已关闭")


def main():
    global current_result, serial_active

    # 清空输出目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化摄像头
    camera_indexes = [0, 1, 2, '/dev/my_camera']
    cap = None
    for index in camera_indexes:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"成功打开摄像头: {index}")
            break
        if cap:
            cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError("无法打开任何摄像头")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 获取图像尺寸
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("无法获取摄像头画面")
    img_size = (frame.shape[1], frame.shape[0])  # (width, height)

    # 加载类别名称
    names = load_class_names()
    print(f"加载的类别名称: {names}")  # 调试输出

    # 启动串口发送线程
    serial_thread = threading.Thread(target=serial_sender)
    serial_thread.daemon = True
    serial_thread.start()

    # 初始拍摄6张照片构建基础字符串
    result_list = []
    for i in range(6):
        print(f"初始拍摄 #{i + 1}/6")
        result_str = detect_single_image(cap, names, img_size)
        print(f"检测结果: {result_str}")  # 调试输出
        result_list.append(result_str)
        current_result = '7'.join(result_list) + '7'
        print(f"当前结果: {current_result}")

    # 30次循环更新结果
    for i in range(100):
        print(f"更新循环 #{i + 1}/30")
        # 拍摄新照片并检测
        new_result = detect_single_image(cap, names, img_size)
        print(f"新检测结果: {new_result}")  # 调试输出

        # 更新结果列表
        if len(result_list) >= 6:
            result_list.pop(0)
        result_list.append(new_result)

        # 更新当前结果
        current_result = '7'.join(result_list) + '7'
        print(f"更新后结果: {current_result}")

    # 清理工作
    serial_active = False
    serial_thread.join(timeout=2.0)
    cap.release()
    print("程序执行完毕")


if __name__ == "__main__":
    main()