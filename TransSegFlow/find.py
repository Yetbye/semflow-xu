import numpy as np
from PIL import Image

def count_pixel_values(image_path):
    # 1. 打开图片
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"找不到文件: {image_path}")
        return
    
    # 将图片转换为 numpy 数组
    img_array = np.array(img)
    
    # 2. 使用 np.unique 统计像素值及其数量
    # return_counts=True 会同时返回元素和对应的出现次数
    unique_values, counts = np.unique(img_array, return_counts=True)
    
    # 3. 打印统计结果
    print(f"--- 掩码图片统计: {image_path} ---")
    print(f"图片分辨率: {img.size[0]} x {img.size[1]}")
    print(f"总像素数: {img_array.size}")
    print("-" * 35)
    
    # 将结果打包成字典并打印
    pixel_counts = dict(zip(unique_values, counts))
    for value, count in pixel_counts.items():
        print(f"像素值: {value:<3} | 包含像素个数: {count}")
        
    return pixel_counts

if __name__ == "__main__":
    # 将这里的路径替换为你的实际 png 文件路径
    image_file = "dataset/GDD/train/mask/5419.png" 
    
    count_pixel_values(image_file)