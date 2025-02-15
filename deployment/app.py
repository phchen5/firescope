import torch
from torch import nn
from torchvision import models

from PIL import Image
import streamlit as st
import os


device = torch.device("cpu")

# load model function
def resume(model, filename):
    model.load_state_dict(torch.load(filename, map_location=device))

#define function to predict custom images
def pred_and_plot_image(model, image_path, transform, device: torch.device=device):
    
    img = Image.open(image_path).convert('RGB')

    image_transform = transform

    model.to(device)
    model.eval()
    with torch.inference_mode():

        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

        target_image_pred_probs = torch.sigmoid(target_image_pred)
        target_image_pred_label = torch.round(target_image_pred_probs)
        
    return target_image_pred_label, target_image_pred_probs



# get pretrained model's weight and transformation
weights = models.MobileNet_V3_Small_Weights.DEFAULT
auto_transforms = weights.transforms()

# setup architecture
model_tl = models.mobilenet_v3_small(weights=weights).to(device)
model_tl.classifier = torch.nn.Sequential(
    nn.Linear(in_features=576, out_features=1024, bias=True),
    nn.Hardswish(),
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1024, out_features=1, bias=True)).to(device)

# load weights from pth
model_path = os.path.join(os.path.dirname(__file__), "best_model_tl.pth")
resume(model_tl, model_path)

st.write("""
         # 🔥 Forest Fire Detection
         """
         )
st.write("This is an image classification web app that predicts whether or not a nature image contains forest fire.")
st.write("Note: The model that powers this app is trained using nature images. Therefore, the model may result in incorrect classifications for non-nature images.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    label, probs = pred_and_plot_image(model_tl, file, auto_transforms, device)
    
    if label == 0:
        st.markdown("""
                    <div style='text-align: center; font-family: impact, sans-serif; font-size: 24px;'>
                    Oh no! This is a fire image! 🔥
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
                    <div style='text-align: center; font-family: impact, sans-serif; font-size: 24px;'>
                    Phew! This is not a fire image! 😌
                    </div>
                    """, unsafe_allow_html=True)
    
    confidence = float(abs(probs - 0.5) * 2)
    formatted_confidence = '{:.2%}'.format(confidence)
    
    st.markdown(f"""
                <div style='text-align: center; font-family: Georgia, serif; font-size: 18px;'>
                Confidence of Prediction: {formatted_confidence}
                </p>
                """, unsafe_allow_html=True)

