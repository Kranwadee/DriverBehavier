import streamlit as st 
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title 
st.title('DriverBehavier Classification')

#Set Header 
st.header('Please upload picture')

#Load Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('mobilenetv3_large_1004.pt', map_location=device)

# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

    if st.button('Prediction'):
        # Prediction class
        predicted_class, prob = pred_class(model, image, class_name)
        
        st.write("## Prediction Result")
        st.write(f"### Predicted Class: {predicted_class}")
        st.write(f"### Probability: {prob*100:.2f}%")
