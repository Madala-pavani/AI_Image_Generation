import streamlit as st
import matplotlib.pyplot as plt
from model import generate_image

st.title("✨ AI Face Generator")
st.write("Click the button to generate a unique AI-generated face!")

if st.button("🚀 Generate Image"):
    img = generate_image()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    st.pyplot(fig)