import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digit Recognition (SVM)",
    page_icon="✍️",
    layout="centered"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    font-weight: bold;
    color: #4CAF50;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f0fdf4;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>✍️ Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Easy • Accurate • Student Friendly (SVM)</p>", unsafe_allow_html=True)

st.divider()

# ---------------- LOAD DATA ----------------
digits = load_digits()
X = digits.data
y = digits.target

# ---------------- DATA INFO ----------------
st.markdown("### 📘 About the Dataset")
st.write(
    "This dataset contains handwritten digit images (0–9). "
    "Each image is **8×8 pixels**, and the computer learns patterns from them."
)

# ---------------- SHOW DIGIT GALLERY ----------------
st.markdown("### 🖼️ Sample Handwritten Digits")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"{digits.target[i]}")
    ax.axis("off")

st.pyplot(fig)

# ---------------- PREPROCESSING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- SVM MODEL ----------------
model = SVC(kernel="rbf", C=10, gamma=0.01)
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- ACCURACY CARD ----------------
st.markdown("### 📊 Model Performance")

st.markdown(
    f"""
    <div class="card">
        <p class="big-font">✅ Accuracy</p>
        <h1>{accuracy*100:.2f}%</h1>
        <p>SVM Algorithm (Simple & Powerful)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- INTERACTIVE PREDICTION ----------------
st.divider()
st.markdown("### 🔢 Try a Prediction")

st.write(
    "Use the slider to select a test image. "
    "The SVM model will predict the handwritten digit."
)

index = st.slider("Select Image Index", 0, len(X_test) - 1)

test_image = X_test[index].reshape(1, -1)
prediction = model.predict(test_image)[0]

fig2, ax2 = plt.subplots()
ax2.imshow(test_image.reshape(8, 8), cmap="gray")
ax2.set_title(f"Predicted Digit: {prediction}", fontsize=16)
ax2.axis("off")

st.pyplot(fig2)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 App Info")
st.sidebar.write("Algorithm: **Support Vector Machine (SVM)**")
st.sidebar.write("Why SVM?")
st.sidebar.write("- Very high accuracy")
st.sidebar.write("- Works well for images")
st.sidebar.write("- Beginner friendly")

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><p style='text-align:center;'>Made with ❤️ for Students using Streamlit & SVM</p>",
    unsafe_allow_html=True
)
