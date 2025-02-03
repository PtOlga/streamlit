import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2
import gdown

## Specify the path to the folder containing the model
#models_dir = os.path.join(os.path.dirname(__file__), 'models')
#model_path = os.path.join(models_dir, 'best_model_rf.joblib')

## Load the model
#try:
#    model = joblib.load(model_path)
#except FileNotFoundError:
#    st.error(f"Model file not found at {model_path}. Please check the path.")
#    st.stop()

# URL файла на Google Drive
url = 'https://drive.google.com/uc?id=1AyPDoibUsYhx1CnFkFouPh_fIy0pXpB5'

# Локальный путь для сохранения файла
model_path = 'best_model_rf.joblib'

# Загрузка файла с Google Drive
@st.cache_data
def load_model_from_drive():
    try:
        st.write("Загрузка модели с Google Drive...")
        gdown.download(url, model_path, quiet=False)
        st.write("Модель загружена успешно.")
        model = joblib.load(model_path)
        st.write("Модель загружена и готова к использованию.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        


# Загрузка модели
model = load_model_from_drive()

# Function to preprocess the drawn image
def preprocess_image(image):
    """
    Preprocess the drawn image to match the MNIST dataset format.
    - Convert to grayscale.
    - Resize to 28x28 pixels.
    - Invert colors (MNIST uses white digits on a black background).
    - Normalize pixel values to the range [0, 1].
    - Apply thresholding to binarize the image.
    - Remove noise using morphological operations.
    - Flatten the image to a 1D array of 784 elements.
    """
    # Convert to grayscale and resize
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image)
    
    # Invert colors (MNIST uses white digits on black background)
    image_array = 255 - image_array
    
    # Apply Gaussian blur to reduce noise
    image_array = cv2.GaussianBlur(image_array, (3, 3), 0)
    
    # Apply adaptive thresholding to binarize the image
    image_array = cv2.adaptiveThreshold(
        image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Normalize pixel values to the range [0, 1]
    image_array = image_array / 255.0
    
    return image_array

# Streamlit app title
st.title("Digit Recognition with MNIST " + model_path)

# Create three columns: left (drawing canvas), middle (model's view), right (results)
col1, col2, col3 = st.columns([1, 1, 1])  # Three equal-width columns

# Left Column: Drawing Canvas
with col1:
    st.write("### 1. Draw a Digit")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fill color (orange with transparency)
        stroke_width=15,  # Stroke width
        stroke_color="#FFFFFF",  # Stroke color (white)
        background_color="#000000",  # Background color (black)
        update_streamlit=True,  # Update Streamlit on every change
        height=280,  # Canvas height
        width=280,  # Canvas width
        drawing_mode="freedraw",  # Allow free drawing
        key="canvas",
    )
    
  

# Middle Column: What the Model Sees
with col2:
    if canvas_result.image_data is not None:
        st.write("### 2. What the Model Sees")
        st.write("This is how the model processes your drawing:")
        
        # Convert the canvas image to PIL format
        drawn_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        # Preprocess the image
        image_array = preprocess_image(drawn_image)
        
        # Display the preprocessed image (what the model sees)
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.axis('off')  # Hide axes
        st.pyplot(fig)

# Right Column: Prediction Results & Confusion Matrix
with col3:
    if canvas_result.image_data is not None:
        st.write("### 3. Prediction Results")
        
        # Predict using the model
        prediction = model.predict(image_array.reshape(1, -1))
        prediction_proba = model.predict_proba(image_array.reshape(1, -1))  # Get probability scores
        confidence = np.max(prediction_proba) * 100  # Calculate confidence percentage

        # Display the prediction results
        st.write(f"**Predicted Digit:** {prediction[0]}")
        st.write(f"**Prediction Confidence:** {confidence:.2f}%")

        # Display the probability distribution for each class
        st.write("**Probability Distribution:**")
        fig, ax = plt.subplots()
        classes = np.arange(10)  # Digits 0-9
        ax.bar(classes, prediction_proba[0], color='skyblue')
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_xticks(classes)
        ax.set_title("Probability for Each Digit")
        st.pyplot(fig)


