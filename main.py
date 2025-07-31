# To run: streamlit run app.py

import streamlit as st
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")

from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from util import BahdanauAttention, ImageCaptionGenerator
from streamlit_cropper import st_cropper

# ------------------------ Load Model & Tokenizer ------------------------ #
@st.cache_resource
def load_model_tokenizer():
    model = load_model(
        "model3.keras",
        custom_objects={
            "ImageCaptionGenerator": ImageCaptionGenerator,
            "BahdanauAttention": BahdanauAttention
        }
    )
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

# ------------------------ Preprocessing ------------------------ #
def preprocess_image(pil_img, target_size=(299, 299)):
    pil_img = pil_img.resize(target_size)
    img_array = img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# ------------------------ UI Layout ------------------------ #
st.markdown("<h1 style='text-align: center;'>🖼️ Smart Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Created by <b>Dayshaun Kakadiya</b> and <b>Rajan Kushwah</b></h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload, Crop, and Caption any image using AI!</p>", unsafe_allow_html=True)

# ------------------------ Upload Image ------------------------ #
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # ------------------------ Rotation ------------------------ #
    # with st.expander("🎛️ Optional: Rotate image"):
    #     rotation = st.slider("Rotate (°)", -180, 180, step=90, value=0)
    #     img = img.rotate(rotation, expand=True)

    # ------------------------ Cropper ------------------------ #
    st.markdown("### ✂️ Crop Image")
    cropped_img = st_cropper(
        img,
        realtime_update=True,
        box_color='#00f',
        aspect_ratio=None,
    )


    # ------------------------ Generate Caption ------------------------ #
    if st.button("🚀 Generate Caption"):
        with st.spinner("Generating caption..."):
            image_tensor = preprocess_image(cropped_img)
            caption = model.predict_caption(image_tensor, tokenizer)
            # ------------------------ Show Cropped Image ------------------------ #
            st.image(cropped_img, caption="🖼️ Final Image")
        st.success("📝 Caption: " + caption)

else:
    st.info("Upload a `.jpg`, `.jpeg`, or `.png` image to get started.")
# ------------------------ Footer ------------------------ #
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        Created by <b>Dayshaun Kakadiya</b> and <b>Rajan Kushwah</b><br><br>
        🔗 <b>LinkedIn:</b>
        <a href='https://www.linkedin.com/in/dayshaun-kakadiya-ba2410321' target='_blank'>Dayshaun</a> |
        <a href='https://www.linkedin.com/in/rajan-kushwah-b4751129a' target='_blank'>Rajan</a><br>
        💻 <b>GitHub:</b>
        <a href='https://github.com/dayshaun-1' target='_blank'>Dayshaun</a> |
        <a href='https://github.com/kushwahrajan20' target='_blank'>Rajan</a>
    </div>
    """,
    unsafe_allow_html=True
)

