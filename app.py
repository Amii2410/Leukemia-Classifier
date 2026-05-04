# ==============================
# IMPORTS
# ==============================
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Leukemia Classifier", layout="wide")

# ==============================
# STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# 🌗 AUTO THEME
# ==============================
dark_mode = st.sidebar.toggle("🌗 Dark Mode", True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Controls")
show_probs = st.sidebar.toggle("Probabilities", True)
show_gradcam = st.sidebar.toggle("Grad-CAM", True)
show_confusion = st.sidebar.toggle("Confusion Matrix", True)
show_history = st.sidebar.toggle("📜 Show History", False)

# ==============================
# CSS
# ==============================
bg_overlay = "rgba(5,5,10,0.85)" if dark_mode else "rgba(255,255,255,0.9)"
text_color = "#00FFFF" if dark_mode else "#0077ff"

st.markdown(f"""
<style>

.stApp {{
    background: linear-gradient({bg_overlay},{bg_overlay}),
    url("https://images.unsplash.com/photo-1579154204601-01588f351e67");
    background-size: cover;
    background-attachment: fixed;
    color: {"white" if dark_mode else "black"};
}}

.title-text {{
    font-size:60px;
    font-weight:800;
    background: linear-gradient(90deg, #00FFFF, #00FFAA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 2s infinite alternate;
}}

@keyframes glow {{
    from {{ text-shadow: 0 0 10px #00FFFF; }}
    to {{ text-shadow: 0 0 30px #00FFAA; }}
}}

.gradcam-title {{
    font-size:28px;
    font-weight:bold;
    text-align:center;
    margin-top:30px;
    margin-bottom:20px;
    color:{text_color};
    animation: glow 2s infinite alternate;
}}

.glass {{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius:15px;
    padding:15px;
    margin-bottom:20px;
}}

.predicted {{
    animation: pulse 1.5s infinite;
    border-radius:10px;
    padding:6px;
}}

@keyframes pulse {{
    0% {{ box-shadow: 0 0 5px #00FFAA; }}
    50% {{ box-shadow: 0 0 20px #00FFAA; }}
    100% {{ box-shadow: 0 0 5px #00FFAA; }}
}}

img {{
    border-radius:12px;
}}

img:hover {{
    transform: scale(1.04);
    transition:0.3s;
}}

</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.markdown("""
<div style="text-align:center;">
    <span style="font-size:60px;">🧬</span>
    <span class="title-text">Leukemia Classifier</span>
</div>
""", unsafe_allow_html=True)

# ==============================
# MODEL
# ==============================
class_names = ['ALL','AML','CLL','CML','Healthy']

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model():
    base = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(5, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)

@st.cache_resource
def load_model():
    m = build_model()
    m.load_weights("leukemia_model.h5")
    return m

model = load_model()

# ==============================
# GRADCAM
# ==============================
def get_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, pred = grad_model(img_array)
        idx = tf.argmax(pred[0])
        loss = pred[:, idx]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv[0]

    heatmap = tf.reduce_sum(conv * pooled, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap)+1e-8)
    return heatmap

# ==============================
# EVALUATION
# ==============================
@st.cache_data
def evaluate(_model):
    gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        "dataset", target_size=(224,224),
        batch_size=32, class_mode='categorical', shuffle=False
    )
    preds = _model.predict(gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = gen.classes
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm)/np.sum(cm)
    return cm, list(gen.class_indices.keys()), acc

# ==============================
# UPLOAD
# ==============================
file = st.file_uploader("📤 Upload Blood Smear Image", type=["jpg","png","jpeg"])

if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8),1)
    img_r = cv2.resize(img,(224,224))
    arr = np.expand_dims(img_r/255.0,0)

    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), width=300)

    # LOADING
    loading = st.empty()
    for t in ["Analyzing...", "Detecting...", "Finalizing..."]:
        loading.markdown(f"### 🔍 {t}")
        time.sleep(0.5)

    pred = model.predict(arr)
    loading.empty()

    pred_class = class_names[np.argmax(pred)]
    conf = float(np.max(pred))

    st.success(f"{pred_class} ({conf*100:.2f}%)")

    # STORE HISTORY
    st.session_state.history.append({
        "class":pred_class,
        "confidence":conf,
        "time":datetime.datetime.now().strftime("%H:%M:%S"),
        "image":img_r.copy()
    })

    # PROBABILITIES
    if show_probs:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Class Probabilities")
        for i,c in enumerate(class_names):
            p=int(pred[0][i]*100)
            if c==pred_class:
                st.markdown(f'<div class="predicted"><b>{c}: {p}%</b></div>', unsafe_allow_html=True)
            else:
                st.write(f"{c}: {p}%")
            st.progress(p)
        st.markdown('</div>', unsafe_allow_html=True)

    # GRADCAM
    if show_gradcam:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="gradcam-title"> Grad-CAM Visualization</div>', unsafe_allow_html=True)

        heat=get_gradcam(model,arr)
        heat=cv2.resize(heat,(224,224))
        heat=np.uint8(255*heat)
        heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET)
        overlay=cv2.addWeighted(img_r,0.7,heat,0.6,0)

        col1,col2=st.columns(2)
        col1.image(cv2.cvtColor(img_r,cv2.COLOR_BGR2RGB))
        col2.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))

        st.markdown('</div>', unsafe_allow_html=True)

    # DOWNLOAD REPORT
    if st.button("📥 Download Report Image"):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis('off')
        ax.text(0.1,0.8,f"Prediction: {pred_class}\nConfidence: {conf*100:.2f}%", fontsize=12)
        fig.savefig("report.png")

        with open("report.png","rb") as f:
            st.download_button("Download", f, "report.png")

# ==============================
# CONFUSION MATRIX
# ==============================
if show_confusion and st.button("Run Evaluation"):
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    cm,labels,acc=evaluate(model)
    fig,ax=plt.subplots(figsize=(4.5,3))
    sns.heatmap(cm,annot=True,fmt="d",cmap="mako",
                xticklabels=labels,yticklabels=labels,
                cbar=False,ax=ax)

    st.pyplot(fig)
    st.success(f"Accuracy: {acc*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# HISTORY
# ==============================
if show_history:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📜 Prediction History")

    for item in reversed(st.session_state.history[-5:]):
        col1,col2=st.columns([1,3])
        col1.image(cv2.cvtColor(item["image"],cv2.COLOR_BGR2RGB),width=80)
        col2.write(f"{item['class']} ({item['confidence']*100:.2f}%) — {item['time']}")

    st.markdown('</div>', unsafe_allow_html=True)