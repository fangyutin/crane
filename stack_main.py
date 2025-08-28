import cv2
import time
import os
import subprocess
import yaml
import shutil
from glob import glob
import serial
from collections import deque

# 配置路径
YOLOV5_DIR = "/home/pi2/yolo/yolov5-master"
OUTPUT_DIR = "/home/pi2/yolo/outcome"
TEMP_DIR = os.path.join(YOLOV5_DIR, "temp_image")
os.makedirs(TEMP_DIR, exist_ok=True)

# 结果文件路径
RESULT_FILE = os.path.join(OUTPUT_DIR, "results.txt")

# 全局变量
result_queue = deque(maxlen=40)  # 存储最近40个结果
current_long_string = ""  # 当前的长字符串
serial_active = True  # 串口发送控制标志


def apply_digital_zoom(frame, zoom_factor=1, center_x=0.5, center_y=0.65):
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


def capture_single_image(cap, warmup_frames=3):
    """拍摄单张图片"""
    # 预热几帧
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("摄像头读取失败")

    # 应用数码变焦
    zoom_frame = apply_digital_zoom(frame)

    # 保存为固定文件名
    image_path = os.path.join(TEMP_DIR, "temp.jpg")
    cv2.imwrite(image_path, zoom_frame)
    return image_path


def load_class_names():
    """加载类别名称"""
    try:
        with open(os.path.join(YOLOV5_DIR, "data", "coco.yaml")) as f:
            data = yaml.safe_load(f)
            return data['names']
    except Exception as e:
        print(f"加载类别名称失败: {e}")
        return []  # 返回空列表避免后续错误


def get_norepeat_string(result_chars):
    # 辅助函数：移除重复（处理字符列表）
    def remove_duplicates(chars):
        # 获取所有数字字符
        digits = [char for char in chars if char.isdigit()]

        # 当存在重复数字时循环处理
        while len(digits) != len(set(digits)):
            found = False
            # 按指定顺序遍历位置
            for idx in [1, 5, 0, 2, 3, 4]:
                if idx >= len(chars) or chars[idx].isalpha():
                    continue  # 跳过字母位置

                # 检查当前数字是否重复
                if chars.count(chars[idx]) > 1:
                    # 获取已存在的数字集合
                    existing_digits = set(d for d in chars if d.isdigit())
                    # 按优先级查找可替换的数字
                    for candidate in ['6', '4', '5', '1', '2', '3']:
                        if candidate not in existing_digits:
                            chars[idx] = candidate
                            found = True
                            break
                    if found:
                        break

            # 更新数字列表
            digits = [char for char in chars if char.isdigit()]
            if not found:
                break

        return chars

    # 主处理逻辑
    # 确保处理的是字符列表
    chars = list(result_chars) if isinstance(result_chars, str) else result_chars[:]

    # 统计字母数量
    letter_count = sum(1 for char in chars if char.isalpha())

    # 情况1：字母数量大于1
    if letter_count > 1:
        # 统计特定字母
        count_abcdef = sum(1 for char in chars if char in 'abcdef')
        count_x = chars.count('x')

        if count_abcdef == 0 and count_x > 0:
            # 情况：abcdef出现次数为0，先替换第一个x为字母
            for i, char in enumerate(chars):
                if char == 'x':
                    # 位置映射：0->a, 1->b,...5->f，其他位置映射到a
                    if i < 6:
                        chars[i] = chr(ord('a') + i)
                    else:
                        chars[i] = 'a'
                    count_abcdef += 1
                    count_x -= 1
                    letter_count = count_abcdef + count_x
                    break

        if letter_count > 1:
            # 重新统计
            count_abcdef = sum(1 for char in chars if char in 'abcdef')
            count_x = chars.count('x')

            # 按指定顺序查找字母位置
            for idx in [1, 5, 0, 2, 3, 4]:
                if idx >= len(chars):
                    continue

                # 情况1: 只存在1个abcdef字母 -> 只替换x
                if count_abcdef == 1:
                    if chars[idx] == 'x':
                        # 获取已存在的数字集合
                        existing_digits = set(char for char in chars if char.isdigit())
                        # 按优先级选择替换数字
                        for candidate in ['6', '4', '5', '1', '2', '3']:
                            if candidate not in existing_digits:
                                chars[idx] = candidate
                                letter_count -= 1
                                count_x -= 1
                                break
                        if letter_count <= 1:
                            break

                # 情况2/3: 存在0个或多个abcdef字母
                else:
                    # 优先替换x
                    if chars[idx] == 'x':
                        existing_digits = set(char for char in chars if char.isdigit())
                        for candidate in ['6', '4', '5', '1', '2', '3']:
                            if candidate not in existing_digits:
                                chars[idx] = candidate
                                letter_count -= 1
                                count_x -= 1
                                break
                        if letter_count <= 1:
                            break

                    # 当没有x时再替换abcdef字母
                    elif chars[idx] in 'abcdef' and count_x == 0:
                        existing_digits = set(char for char in chars if char.isdigit())
                        for candidate in ['6', '4', '5', '1', '2', '3']:
                            if candidate not in existing_digits:
                                chars[idx] = candidate
                                letter_count -= 1
                                count_abcdef -= 1
                                break
                        if letter_count <= 1:
                            break

    # 情况2：字母数量等于1
    if letter_count == 1:
        for i, char in enumerate(chars):
            if char == 'x':
                if i < 6:
                    chars[i] = chr(ord('a') + i)
                else:
                    chars[i] = 'a'
        chars = remove_duplicates(chars)

    # 情况3：没有字母
    elif letter_count == 0:
        # 检查数字是否有重复
        if len(set(chars)) == len(chars):
            # 无重复：将第二个字符改为'b'
            if len(chars) > 1:
                chars[1] = 'b'
        else:
            # 有重复：按顺序查找第一个重复数字位置
            for idx in [1, 5, 0, 2, 3, 4]:
                if idx >= len(chars):
                    continue

                if chars.count(chars[idx]) > 1:
                    # 将该位置替换为'b'
                    chars[idx] = 'b'
                    break
            chars = remove_duplicates(chars)

    return chars


def get_detected_string(label_path, names):
    """从单个标签文件获取检测结果字符串"""
    if not label_path or not os.path.exists(label_path):
        return "error"

    try:
        # 定义六个区域的边界（x_min, x_max, y_min, y_max）
        # 根据您的实际需求调整这些值
        region_bounds = [
            # 区域 a
            (0, 0.1, 0.5, 0.764),
            # 区域 b
            (0.133, 0.234, 0.222, 0.402),
            # 区域 c
            (0.289, 0.406, 0.22, 0.417),
            # 区域 d
            (0.46, 0.587, 0.194, 0.403),
            # 区域 e
            (0.625, 0.738, 0.167, 0.33),
            # 区域 f
            (0.728, 0.89, 0.44, 0.68)
        ]
        # 初始化区域检测结果
        region_detected = [False] * 6
        region_chars = [''] * 6

        # 读取所有检测结果并解析坐标
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])

                # 确保类别名称是字符串
                class_name = str(names[class_id]) if class_id < len(names) else str(class_id)

                # 检查检测结果是否在任一区域内
                for i, (x_min, x_max, y_min, y_max) in enumerate(region_bounds):
                    if (x_min <= x_center < x_max) and (y_min <= y_center < y_max):
                        region_detected[i] = True
                        region_chars[i] = class_name
                        # 找到匹配区域后跳出循环
                        break

        # 生成最终字符列表
        result_chars = []
        for i in range(6):
            if region_detected[i]:
                detected_char = region_chars[i]
                # 只接受字母a-f或数字1-6
                if detected_char in 'abcdef123456':
                    result_chars.append(detected_char)
                else:
                    result_chars.append(chr(ord('a') + i))
            else:
                result_chars.append('x')

        # 处理重复问题
        result_chars = get_norepeat_string(result_chars)
        result = ''.join(result_chars)
        return result

    except Exception as e:
        print(f"处理标签文件失败: {e}")
        return "error"


def run_yolo_detection(image_path):
    """运行YOLOv5检测并返回标签路径"""
    try:
        # 清空输出目录
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 运行检测
        subprocess.run([
            '/home/pi2/PycharmProjects/stack_test/.venv/bin/python',
            '/home/pi2/yolo/yolov5-master/detect.py',
            '--source', image_path,
            '--project', OUTPUT_DIR,
            '--name', 'detections',
            '--exist-ok',
            '--save-txt',
            '--save-conf'
        ], cwd=YOLOV5_DIR, check=True)

        # 返回标签路径
        return os.path.join(OUTPUT_DIR, "detections", "labels", "temp.txt")

    except Exception as e:
        print(f"YOLO检测失败: {e}")
        return None


def save_results_to_file():
    """将结果队列保存到文件"""
    try:
        with open(RESULT_FILE, "w") as f:
            for result in result_queue:
                f.write(str(result) + "\n")  # 确保写入的是字符串
    except Exception as e:
        print(f"保存结果失败: {e}")


def load_results_from_file():
    """从文件加载结果队列"""
    global result_queue
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r") as f:
                # 读取所有行，过滤空行，并确保是字符串
                lines = [line.strip() for line in f.readlines() if line.strip()]
                result_queue = deque(lines, maxlen=40)
                print(f"从文件加载了 {len(lines)} 个结果")
        except Exception as e:
            print(f"加载结果失败: {e}")
            result_queue = deque(maxlen=40)


def serial_sending_thread(ser):
    """串口发送线程函数"""
    global current_long_string, serial_active

    print("串口发送线程启动")
    while serial_active:
        try:
            if current_long_string:
                # 确保是字符串
                message = str(current_long_string) + '\n'
                ser.write(message.encode())
                print(f"已发送: {message.strip()}")
            time.sleep(1)  # 每秒发送一次
        except Exception as e:
            print(f"串口发送错误: {e}")
            time.sleep(2)  # 出错后等待2秒再重试

    print("串口发送线程停止")


def main():
    global current_long_string, serial_active, result_queue

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 尝试加载之前的结果
    load_results_from_file()

    # 初始化摄像头
    cap = None
    for camera_index in ['/dev/my_camera', 0, 1, 2, 3, 4, 5]:
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # 尝试读取一帧测试
                ret, frame = cap.read()
                if ret:
                    print(f"成功打开摄像头: {camera_index}")
                    break
            cap.release()
        except:
            continue

    if cap is None or not cap.isOpened():
        # 所有尝试都失败
        file_path = os.path.join(OUTPUT_DIR, "fail.txt")
        with open(file_path, 'w') as file:
            file.write("无法打开摄像头")
        print(f"文件 {file_path} 已创建")
        raise RuntimeError("无法获取摄像头画面")

    # 加载类别名称
    names = load_class_names()

    # 初始化串口
    ser = None
    try:
        ser = serial.Serial(
            port='/dev/serial0',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        # 启动串口发送线程
        import threading
        serial_thread = threading.Thread(target=serial_sending_thread, args=(ser,), daemon=True)
        serial_thread.start()
        print("串口发送线程已启动")
    except Exception as e:
        print(f"串口初始化失败: {e}")
        ser = None

    try:
        # 初始阶段：如果结果队列不足6个，拍摄照片直到有6个结果
        while len(result_queue) < 6:
            print(f"初始阶段: 已有 {len(result_queue)} 个结果，需要至少6个")

            # 拍摄单张照片
            image_path = capture_single_image(cap)

            # 运行YOLOv5检测
            label_path = run_yolo_detection(image_path)

            # 获取检测结果字符串
            result_str = get_detected_string(label_path, names) if label_path else "error"
            result_queue.append(result_str)
            print(f"新结果: {result_str}")

            # 保存结果到文件
            save_results_to_file()

            # 更新长字符串（如果已有至少6个结果）
            if len(result_queue) >= 6:
                recent_results = list(result_queue)[-6:]
                # 确保所有元素都是字符串
                recent_results = [str(x) for x in recent_results]
                current_long_string = '7'.join(recent_results) + '7'
                print(f"初始长字符串: {current_long_string}")

            time.sleep(0.5)

        # 持续拍摄和处理
        while True:
            # 拍摄单张照片
            image_path = capture_single_image(cap)

            # 运行YOLOv5检测
            label_path = run_yolo_detection(image_path)

            # 获取检测结果字符串
            result_str = get_detected_string(label_path, names) if label_path else "error"
            result_queue.append(result_str)
            print(f"新结果: {result_str}")

            # 保存结果到文件
            save_results_to_file()

            # 获取最近6个结果
            recent_results = list(result_queue)[-6:]
            # 确保所有元素都是字符串
            recent_results = [str(x) for x in recent_results]

            # 形成新的长字符串
            current_long_string = '7'.join(recent_results) + '7'
            print(f"更新长字符串: {current_long_string}")

            time.sleep(0.5)  # 循环间隔

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        # 打印更多调试信息
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        serial_active = False
        if ser and ser.is_open:
            ser.close()
        if cap and cap.isOpened():
            cap.release()
        print("资源已释放")


if __name__ == "__main__":
    main()