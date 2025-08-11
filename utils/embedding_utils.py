# utils/embedding_utils.py
import torch
from PIL import Image

def get_text_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def get_image_embedding(image_path, transform, model, device):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=img_tensor)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except:
        return None