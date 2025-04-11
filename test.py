from ultralytics import YOLO

# 加载训练好的最佳模型
model = YOLO('runs/detect/train/weights/best.pt')

# 在新图像上进行预测
results = model(r'D:\Download\git\GitWarehouse\turningVolov8\ultralytics\datasets\mydata\images\0E6DB6EB85D6CB7E5B5B88BB48650077.jpg')

# 显示结果
results[0].show()