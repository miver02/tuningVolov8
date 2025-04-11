# 停机坪检测系统 (Helipad Detection System)

基于YOLOv8的直升机停机坪(Helipad)目标检测模型训练与应用。

## 项目概述

本项目利用YOLOv8目标检测算法，对直升机停机坪进行自动识别和定位。停机坪通常以"H"形标记表示，模型经过训练后能够在复杂背景下准确检测出停机坪位置，为无人机、直升机自主着陆提供视觉引导。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- Ultralytics YOLOv8
- 其他依赖见requirements.txt（如有）

## 数据集结构

数据集采用Labelme标注工具进行标注，原始数据组织如下：

```
ultralytics/datasets/mydata/
├── images/              # 原始图像文件夹
│   ├── *.jpg
│   ├── *.jpeg
│   ├── *.png
│   └── *.webp
├── Annotations/         # Labelme格式的JSON标注文件
│   └── *.json
├── labels/              # 转换后的YOLO格式标注
│   └── *.txt
├── train.txt            # 训练集图像路径列表
└── val.txt              # 验证集图像路径列表
```

标注标签：
- `H`: 停机坪 (类别索引: 0)

## 数据预处理

项目包含从Labelme格式转换为YOLO格式的转换脚本：

```
ultralytics/datasets/mydata/convert_labelme_to_yolo.py
```

主要转换步骤：
1. 读取Labelme格式的JSON标注文件
2. 提取矩形标注框坐标
3. 转换为YOLO格式（归一化的中心点坐标+宽高）
4. 生成train.txt和val.txt文件列表

## 模型配置

模型配置文件位于：`ultralytics/datasets/shu.yaml`

```yaml
path: D:/Download/git/GitWarehouse/turningVolov8/ultralytics/datasets/mydata  # 数据集根目录
train: D:/Download/git/GitWarehouse/turningVolov8/ultralytics/datasets/mydata/train.txt
val: D:/Download/git/GitWarehouse/turningVolov8/ultralytics/datasets/mydata/val.txt

# 数据集类别
names:
  0: helipad  # 停机坪
```

## 训练过程

### 训练命令

```bash
yolo train data=ultralytics/datasets/shu.yaml model=yolov8n.pt epochs=100 lr0=0.01
```

### 训练参数

- 基础模型: YOLOv8n (nano版本)
- 训练轮次: 100 epochs
- 批次大小: 16
- 学习率: 0.01
- 图像大小: 640x640
- 优化器: 自动选择 (Auto)
- 数据增强: 随机水平翻转、mosaic等标准YOLOv8增强方法

### 训练结果

训练过程生成的文件保存在：`runs/detect/train/`目录下
- `weights/best.pt`: 性能最佳的模型权重
- `weights/last.pt`: 最后一个epoch的模型权重
- `results.csv`: 训练过程中的指标记录
- 其他可视化文件: labels.jpg、train_batch*.jpg等

## 模型性能

根据训练记录，模型在验证集上的性能：
- mAP50: ~0.9 (IoU=0.5时的平均精度)
- mAP50-95: ~0.75 (IoU从0.5到0.95的平均精度)
- 精确率(Precision): 接近1
- 召回率(Recall): ~0.5

## 推理与部署

### 使用模型进行推理
#### 详情见test.py
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 在图像上进行预测
results = model('path/to/image.jpg')

# 显示结果
results[0].show()
```

### 模型导出

可以将模型导出为不同格式进行部署：

```bash
# 导出为ONNX格式
yolo export model=runs/detect/train/weights/best.pt format=onnx

# 导出为TensorRT格式
yolo export model=runs/detect/train/weights/best.pt format=engine
```

## 注意事项

1. 在实际使用时，建议根据部署平台的需要，选择合适的模型格式。
2. 当前数据集中训练集和验证集相同，实际应用中应将数据集分成不同部分。
3. 如需提高模型性能，可以考虑：
   - 收集更多的停机坪样本数据
   - 使用更大的基础模型（如yolov8m.pt、yolov8l.pt）
   - 调整学习率和训练更多轮次
   - 增加更多数据增强策略

## 许可证

本项目遵循[项目许可证]。使用YOLOv8请遵循Ultralytics的AGPL-3.0许可证。
