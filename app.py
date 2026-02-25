import openai
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

openai.api_key = "your-openai-api-key"

def analyze_text(text):
    """使用 OpenAI API 分析创意文本"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "请对用户的创意文本进行分析并输出关键词、目的、潜在问题、建议"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message["content"]

def process_sketch(img):
    """处理草图图像"""
    img = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 测试代码
description = "这个设计是一个智能手表的创意。"
result_text = analyze_text(description)

print("文本分析结果:", result_text)

# 示例草图处理
uploaded_image = "your_image_path_here"  # 用上传的图片路径替换
sketch_img = Image.open(uploaded_image)
edges = process_sketch(sketch_img)

# 显示草图边缘提取结果
plt.imshow(edges, cmap="gray")
plt.title("草图边缘提取")
plt.axis("off")
plt.show()
