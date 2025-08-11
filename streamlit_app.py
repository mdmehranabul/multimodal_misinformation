## ✅ `streamlit_app.py`

#```python
import streamlit as st
import torch
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn as nn
import pandas as pd
# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT and ViT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device).eval()

# Define transform for images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=vit_feature_extractor.image_mean, std=vit_feature_extractor.image_std)
])

# Define Hybrid Fusion model
class HybridFusionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6):
        super(HybridFusionClassifier, self).__init__()
        self.text_fc = nn.Linear(input_dim, hidden_dim)
        self.image_fc = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, text_feat, image_feat):
        text_proj = self.text_fc(text_feat)
        image_proj = self.image_fc(image_feat)
        combined = torch.cat((text_proj, image_proj), dim=1)
        return self.classifier(combined)

# Streamlit UI
st.title("Multimodal Misinformation Detection")
st.markdown("Upload an image and enter a caption to classify whether the post is Real or Misleading.")

# Classification type
label_type = st.radio("Choose classification type", ['2-Way', '6-Way'])

# Label map and model selection
if label_type == '6-Way':
    label_map = {
        0: 'Real',
        1: 'Satire',
        2: 'Misleading',
        3: 'Manipulated',
        4: 'False Connection',
        5: 'Imposter Content'
    }
    output_dim = 6
    model_path = "saved_models/hybrid_fusion_6_way_label_name.pt"
else:
    label_map = {
        0: 'Misleading',
        1: 'Real'
    }
    output_dim = 2
    model_path = "saved_models/hybrid_fusion_2_way_label_name.pt"

# Load trained model
model = HybridFusionClassifier(output_dim=output_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Input form
caption = st.text_input("Enter the post caption")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if caption and uploaded_file:
    try:
        # Image embedding
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ Updated here
        img_tensor = image_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feat = vit_model(pixel_values=img_tensor).last_hidden_state[:, 0, :]

        # Text embedding
        inputs = bert_tokenizer(caption, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            text_feat = bert_model(**inputs).last_hidden_state[:, 0, :]

        # Prediction
        with torch.no_grad():
            logits = model(text_feat, image_feat)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            predicted_class = np.argmax(probs)

        st.subheader(f"Prediction: {label_map[predicted_class]}")
        label_names = [label_map[i] for i in range(len(label_map))]
        probs_percent = np.round(probs * 100).astype(int)
        prob_series = pd.Series(probs_percent, index=label_names)
        st.bar_chart(prob_series)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Please enter both a caption and an image.")