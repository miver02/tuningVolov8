#!/usr/bin/env python
# 将Labelme标注格式转换为YOLO格式
import os
import json
import glob
from pathlib import Path

print("当前工作目录:", os.getcwd())

# 配置绝对路径
BASE_DIR = "D:/Download/git/GitWarehouse/turningVolov8/ultralytics/datasets/mydata"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")  # YOLO格式标签将保存在这里

# 打印目录信息
print(f"基础目录: {BASE_DIR}")
print(f"标注目录: {ANNOTATIONS_DIR}")
print(f"图像目录: {IMAGES_DIR}")
print(f"标签目录: {LABELS_DIR}")

# 确保标签目录存在
os.makedirs(LABELS_DIR, exist_ok=True)

# 标签映射 - 根据实际情况修改
# 在JSON中的标签名称到YOLO类别索引的映射
LABEL_MAP = {
    "H": 0,  # 停机坪(helipad)标记为类别0
    "helipad": 0
}

# 检查目录是否存在
if not os.path.exists(ANNOTATIONS_DIR):
    print(f"错误: 标注目录 {ANNOTATIONS_DIR} 不存在!")
else:
    print(f"标注目录 {ANNOTATIONS_DIR} 存在")
    
if not os.path.exists(IMAGES_DIR):
    print(f"错误: 图像目录 {IMAGES_DIR} 不存在!")
else:
    print(f"图像目录 {IMAGES_DIR} 存在")

# 处理所有JSON文件
json_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))
print(f"找到 {len(json_files)} 个JSON文件...")

for json_file in json_files:
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图像尺寸
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # 获取图像文件名（不带扩展名）
    img_filename = os.path.splitext(data['imagePath'])[0]
    
    # 创建对应的标签文件
    txt_path = os.path.join(LABELS_DIR, f"{img_filename}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 处理每个形状（标注）
        for shape in data['shapes']:
            # 获取标签
            label = shape['label']
            
            # 将标签名称映射到YOLO类别索引
            if label in LABEL_MAP:
                class_idx = LABEL_MAP[label]
            else:
                print(f"警告: 未知标签 '{label}' 在文件 {json_file} 中，跳过...")
                continue
            
            # 获取边界框坐标
            if shape['shape_type'] == 'rectangle':
                # Labelme矩形格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # 或者 [[x1,y1], [x2,y2]] (两个对角点)
                points = shape['points']
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                else:  # 四个点的情况
                    x1, y1 = points[0]
                    x2, y2 = points[2]  # 对角点
                
                # 转换为YOLO格式: <class_idx> <center_x> <center_y> <width> <height>
                # 所有值都归一化到0-1范围
                center_x = (x1 + x2) / (2 * img_width)
                center_y = (y1 + y2) / (2 * img_height)
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # 写入YOLO格式
                f.write(f"{class_idx} {center_x} {center_y} {width} {height}\n")
            else:
                print(f"暂不支持的标注类型 '{shape['shape_type']}' 在文件 {json_file} 中，跳过...")

print("转换完成！标签文件已保存到 labels 目录。")

# 创建train.txt和val.txt
# 这里简单地将所有图像都用于训练和验证（实际应用中应该分开）
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))

# 获取绝对路径
image_files = [os.path.abspath(img) for img in image_files]

print(f"找到 {len(image_files)} 个图像文件")

# 写入train.txt
with open(os.path.join(BASE_DIR, "train.txt"), "w", encoding="utf-8") as f:
    for img_path in image_files:
        f.write(f"{img_path}\n")

# 将相同内容写入val.txt（实际应用中应该分开）
with open(os.path.join(BASE_DIR, "val.txt"), "w", encoding="utf-8") as f:
    for img_path in image_files:
        f.write(f"{img_path}\n")

print("创建了train.txt和val.txt文件")