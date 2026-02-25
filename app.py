from flask import Flask, render_template, request
import openai
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

openai.api_key = "your-openai-api-key"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        description = request.form["description"]
        uploaded_file = request.files["sketch"]
        
        # 分析文本
        result_text = analyze_text(description)

        # 处理草图
        sketch_img = Image.open(uploaded_file)
        edges = process_sketch(sketch_img)

        # 显示草图结果（这部分可以在网页上显示）
        return render_template("index.html", result_text=result_text, edges=edges)

    return render_template("index.html")

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

if __name__ == "__main__":
    app.run(debug=True)
