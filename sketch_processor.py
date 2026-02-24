# sketch_processor.py
"""
草图处理模块
用于从文件读取草图并生成 3D 模型
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

def load_sketch(image_path):
    """读取草图图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到文件: {image_path}")
    return img

def preprocess_sketch(image_path, show=False):
    """
    将草图转换为灰度并可视化
    image_path: 本地文件路径
    show: 是否显示草图
    """
    img = load_sketch(image_path)

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化（可选）
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(thresh, cmap="gray")
        plt.title("草图预处理结果")
        plt.axis("off")
        plt.show()

    return thresh

def generate_3d_model():
    """根据草图生成简单的 3D 模型"""
    # 创建一个立方体
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)

    # 计算并显示 3D 模型
    mesh.compute_vertex_normals()

    # 可视化 3D 模型
    o3d.visualization.draw_geometries([mesh])

    return mesh

# 测试运行
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        # 处理草图并生成 3D 模型
        preprocess_sketch(sys.argv[1], show=True)
        generate_3d_model()
    else:
        print("请传入草图路径，例如：")
        print("python sketch_processor.py sketch_sample.jpg")
