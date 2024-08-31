import streamlit as st 
from PIL import Image
from prediction import pred_class
import torch

# Set title 
st.title('Driver Behavior Classification')

# Set Header 
st.header('Please upload a picture')

# Load Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('mobilenetv3_large_1004.pt', map_location=device)
model.eval()

# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

    if st.button('Predict'):
        try:
            # Prediction class
            predicted_class, prob = pred_class(model, image, class_name)
            
            st.write("## Prediction Result")
            st.write(f"**Class:** {predicted_class}")
            st.write(f"**Probability:** {prob * 100:.2f}%")
        
        except Exception as e:
            st.error(f"Error occurred during prediction: {str(e)}")
