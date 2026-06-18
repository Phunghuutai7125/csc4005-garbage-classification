"""
app.py – Demo Streamlit cho Garbage Classification
Chạy: streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# Config
ONNX_PATH       = "outputs/models/best_model.onnx"
CLASS_TO_IDX    = "outputs/cnn_baseline_finetune/class_to_idx.json"
IMG_SIZE        = 224

CLASS_EMOJI = {
    "cardboard": "📦",
    "glass":     "🍶",
    "metal":     "🥫",
    "paper":     "📄",
    "plastic":   "🧴",
    "trash":     "🗑️",
}

DISPOSAL_TIPS = {
    "cardboard": "Gấp phẳng, bỏ vào thùng rác tái chế màu xanh.",
    "glass":     "Rửa sạch, bỏ vào thùng rác tái chế. Tránh vỡ.",
    "metal":     "Rửa sạch, bỏ vào thùng tái chế kim loại.",
    "paper":     "Giữ khô ráo, bỏ vào thùng tái chế giấy.",
    "plastic":   "Rửa sạch, kiểm tra ký hiệu tái chế trước khi bỏ.",
    "trash":     "Bỏ vào thùng rác thông thường.",
}


@st.cache_resource
def load_model():
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    with open(CLASS_TO_IDX) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return sess, idx_to_class


def preprocess(img: Image.Image) -> np.ndarray:
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf(img.convert("RGB")).unsqueeze(0).numpy()


def predict(sess, idx_to_class, img: Image.Image):
    inp = preprocess(img)
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    probs = np.exp(out[0]) / np.exp(out[0]).sum()
    top3  = np.argsort(probs)[::-1][:3]
    return [(idx_to_class[i], float(probs[i])) for i in top3]


# ---- UI ----
st.set_page_config(page_title="Garbage Classifier", page_icon="♻️", layout="centered")
st.title("♻️ Hệ thống phân loại rác thải")
st.markdown("Upload ảnh rác thải, mô hình sẽ phân loại và gợi ý cách xử lý.")

try:
    sess, idx_to_class = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Không load được model ONNX: {e}\nHãy chạy export_onnx.py trước!")
    model_loaded = False

uploaded = st.file_uploader("Chọn ảnh rác thải", type=["jpg", "jpeg", "png"])

if uploaded and model_loaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Ảnh đầu vào", use_column_width=True)

    with col2:
        with st.spinner("Đang phân tích..."):
            results = predict(sess, idx_to_class, img)

        top_cls, top_prob = results[0]
        emoji = CLASS_EMOJI.get(top_cls, "♻️")
        tip   = DISPOSAL_TIPS.get(top_cls, "")

        st.markdown(f"### {emoji} {top_cls.upper()}")
        st.progress(top_prob)
        st.markdown(f"**Độ tin cậy:** {top_prob:.1%}")
        st.info(f"💡 **Cách xử lý:** {tip}")

        st.markdown("#### Top-3 dự đoán")
        for cls, prob in results:
            st.markdown(f"- {CLASS_EMOJI.get(cls,'♻️')} **{cls}**: {prob:.1%}")

st.markdown("---")
st.markdown("CSC4005 – Học sâu | Nhóm KHMT 17-01")
