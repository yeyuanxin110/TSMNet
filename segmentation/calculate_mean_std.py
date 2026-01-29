import numpy as np
import cv2
import tifffile
import os
def calculate_mean_std(image_paths):
    means, stds = [], []
    for image_path in image_paths:
        img = tifffile.imread(image_path)
        #img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取4波段图像
        img = img.astype(np.float32)
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean, std

# 假设你有一个包含4波段图像路径的列表
#image_paths = ['E:\daijinkun\SegEarth-OV\SegEarth-OV-main\data\SWJTU\img_optical_sar/val/1_1.tif']

directory = r'E:\daijinkun\SegEarth-OV\SegEarth-OV-main\data\SWJTU\img_optical_sar\val'
image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tif')]
mean, std = calculate_mean_std(image_paths)
print(f"Mean: {mean}")
print(f"Std: {std}")