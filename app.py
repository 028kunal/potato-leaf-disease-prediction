# import streamlit as st
# import tensorflow as tf
# from PIL import Image   
# import numpy as np

# #Loading of the model 
# prod_model = tf.keras.models.load_model("saved_models/1.keras")
# beta_model = tf.keras.models.load_model("saved_models/2.keras")

# CLASS_NAMES = ["Early Bright", "Late Blight", "Healthy"]

# st.markdown(
#     "<h1 style='font-size: 60px; font-weight: 700; margin-bottom: 0.5rem;'>Potato Leaf Disease Predictor</h1>",
#     unsafe_allow_html=True
# )

# st.write("Upload a image of a potato leaf to predict if it has Early Blight, Late Blight, or is Healthy.")

# #Upload File
# uploaded_file = st.file_uploader("Choose an image...",type=["jpg","png","jpeg"])

# if uploaded_file is not None:

#     #Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', width=200)


#     #Preprocessing of the Image
#     img_array = np.array(image)
#     img_batch = np.expand_dims(img_array, 0)

#     #Prediction using the production model
#     prediction = prod_model.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
#     confidence = np.max(prediction[0])

#     st.markdown(
#         "<h1 style='font-size: 40px; font-weight: 700; margin-bottom: 0.5rem;'>Model Prediction</h1>",
#         unsafe_allow_html=True
#     )

    
#     st.markdown(
#         f"""
#         <div style="margin-top: 0.5rem; padding: 0.6rem 0.9rem; border-radius: 0.6rem;
#                     background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.5);">
#             <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.2rem;">
#                 Prediction
#             </div>
#             <div style="font-size: 1rem; font-weight: 600; color: #e5e7eb;">
#                 Class: {predicted_class}
#             </div>
#             <div style="font-size: 0.9rem; color: #e5e7eb; margin-top: 0.15rem;">
#                 Confidence: {confidence*100:.2f}%
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page layout: use full width
st.set_page_config(layout="wide")

# Loading of the model
prod_model = tf.keras.models.load_model("saved_models/1.keras")
beta_model = tf.keras.models.load_model("saved_models/2.keras")

CLASS_NAMES = ["Early Bright", "Late Blight", "Healthy"]

# Reduce side padding so content uses more width
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    "<h1 style='font-size: 60px; font-weight: 700; margin-bottom: 0.8rem;'>"
    "Potato Leaf Disease Predictor</h1>",
    unsafe_allow_html=True
)

st.write(
    "Upload an image of a potato leaf to predict if it has Early Blight, "
    "Late Blight, or is Healthy."
)

# Upload File
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Use columns so image and prediction are on one line
if uploaded_file is not None:
    col_img, col_pred = st.columns([1, 1])

    with col_img:
        image = Image.open(uploaded_file)

        # Center the image inside the column
        st.markdown(
            """
            <div style='display: flex; justify-content: center;'>
            """,
            unsafe_allow_html=True
        )
        st.image(image, caption="Uploaded Image", width=240)
        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True
        )

        # Preprocessing of the Image
        img_array = np.array(image)
        img_batch = np.expand_dims(img_array, 0)

        # Prediction using the production model
        prediction = prod_model.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

    with col_pred:
        st.markdown(
            "<h2 style='font-size: 40px; font-weight: 700; margin-bottom: 0.5rem;'>"
            "Model Prediction</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="margin-top: 0.3rem; padding: 0.7rem 1rem; border-radius: 0.6rem;
                        background: rgba(15,23,42,0.9);
                        border: 1px solid rgba(148,163,184,0.5);">
                <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.2rem;">
                    Prediction
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #e5e7eb;">
                    Class: {predicted_class}
                </div>
                <div style="font-size: 0.95rem; color: #e5e7eb; margin-top: 0.2rem;">
                    Confidence: {confidence*100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
