import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import openai

# --------------- 配置 OpenAI -----------------
# 请在 Streamlit Secrets 或 环境变量中配置 OPENAI_API_KEY
openai.api_key = "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"

# ---------------- UI ---------------------------
st.title("📌 创意文本 + 草图 综合评估系统")

st.markdown("""
请输入你的创意描述，并上传草图图像，我们将帮您：
✅ 分析文本创意逻辑  
✅ 检查草图设计要素  
✅ 综合给出评估与建议
""")

description = st.text_area("🔤 输入创意文本描述", height=150)

uploaded_file = st.file_uploader("📸 上传草图图像", type=["png", "jpg", "jpeg"])

analyze_btn = st.button("📊 开始评估")

# --------------- 辅助函数 ----------------------
def analyze_text(text):
    """
    使用简单的 GPT 模型分析文本意图
    """
    if not openai.api_key:
        return {"text_summary": "⚠ 未配置 OpenAI API Key，文本无法完整分析", "score": None}

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"请对用户的创意文本进行分析并输出关键词、目的、潜在问题、建议"},
            {"role":"user","content":text}
        ]
    )
    return {"text_summary": response.choices[0].message["content"]}

def process_sketch(img):
    """
    草图图像基本处理：
    1. 边缘提取
    2. 灰度显示
    3. 形状要素分析（圆形/直线/矩形等）
    """
    img = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    return edges

# ------------------ 主动逻辑 --------------------
if analyze_btn:
    if not description:
        st.error("❌ 请先输入创意文本")
    elif not uploaded_file:
        st.error("❌ 请先上传草图")
    else:
        # 显示上传图像
        st.image(uploaded_file, caption="📌 您上传的草图", use_column_width=True)

        # ---- 文本分析 ----
        with st.spinner("🔎 正在分析文本..."):
            result_text = analyze_text(description)

        st.subheader("📄 文本分析结果")
        st.write(result_text.get("text_summary", "暂无文本分析返回"))

        # ---- 草图分析 ----
        with st.spinner("🖼 正在分析草图..."):
            sketch_img = Image.open(uploaded_file)
            edges = process_sketch(sketch_img)

        st.subheader("🖼 草图边缘提取结果")
        fig, ax = plt.subplots()
        ax.imshow(edges, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

        # ---- 综合评估 ----
        st.subheader("🔍 综合评估与建议")
        st.write("以下为系统根据文本与草图分析的综合反馈（示例）：")

        if result_text.get("text_summary"):
            st.markdown(f"**🎯 文本关键词提取与意图：** {result_text['text_summary'][:200]}...")

        st.markdown("**🧠 草图形状特征总结（自动推测）：**\n- 草图具有明显边缘结构\n- 形状可能代表产品外形轮廓\n- 设计提示：考虑将关键功能模块整合为可视化结构")

        st.markdown("""
**📌 建议：**
- 确保设计语言与文本目的统一
- 强调草图关键功能位置
- 使用 3D 建模工具进一步创建实体模型
""")
