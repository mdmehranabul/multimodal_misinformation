
# 📚 Multimodal Misinformation Detection

This project is a dissertation-level solution for detecting **misinformation in social media posts** using both **text and image modalities**. It leverages deep learning with **BERT** and **Vision Transformer (ViT)** models, using different **fusion strategies** to combine modalities and supports both **2-way** and **6-way** classification.

---

## 🚀 Features

- **Text encoder**: BERT (`bert-base-uncased`)
- **Image encoder**: Vision Transformer (`vit-base-patch16-224`)
- **Fusion models supported**:
  - Early Fusion
  - Late Fusion
  - Hybrid Fusion
- **Supports both**:
  - ✅ 2-Way classification (Real vs Misleading)
  - ✅ 6-Way classification (Real, Satire, Misleading, Manipulated, Fake Connection, Imposter Content)
- Includes:
  - Univariate & Multivariate analysis
  - Fully functional **Streamlit app** for end-user predictions

---

## 🗂️ Project Structure

```
.
├── main.py                      # Training + evaluation script
├── models/
│   ├── fusion_models.py         # Model architectures (early, late, hybrid)
│   ├── dataset.py               # Multimodal dataset class
│   └── training.py              # Training and evaluation utilities
├── utils/
│   ├── preprocessing.py         # Data cleaning, download, image prep
│   ├── embedding_utils.py       # Text & image embedding generation
│   └── analysis.py              # Label distribution & plotting
├── streamlit_app.py             # Streamlit web app to test predictions
├── saved_models/
│   ├── hybrid_fusion_2_way_label_name.pt
│   └── hybrid_fusion_6_way_label_name.pt
├── data/
│   └── multimodal_test_public.tsv
├── requirements.txt
└── README.md
```

---

## 📊 Exploratory Analysis

Included inside `utils/analysis.py`, you’ll find:

- 📌 Distribution of 2-way and 6-way labels
- 📌 Number of posts over time
- 📈 Countplots and histograms using Seaborn

---

## 🗃️ Dataset

Dataset used:
- **File**: `data/multimodal_test_public.tsv`
- Columns used: `title`, `image_url`, `2_way_label`, `6_way_label`, `created_utc`

---

## 🧠 Training & Evaluation

To train and evaluate all models across both 2-way and 6-way tasks:

```bash
python main.py
```

This will:
- Train Early, Late, and Hybrid fusion models for both label sets
- Print accuracy and F1-score comparisons
- Save the best `Hybrid` models to `saved_models/` for use in the Streamlit app

---

## 🌐 Streamlit App

Use this app to test predictions by uploading an image and entering a caption.

### Start the app:

```bash
streamlit run streamlit_app.py
```

### Features:
- Upload post image
- Enter caption or post title
- Choose between 2-way and 6-way classification
- View predicted label + confidence chart

---

## 🧰 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Key libraries:

- `transformers`
- `torch`
- `streamlit`
- `scikit-learn`
- `pandas`, `matplotlib`, `seaborn`

---

## 📌 Notes

- Make sure `saved_models/` contains:
  - `hybrid_fusion_2_way_label_name.pt`
  - `hybrid_fusion_6_way_label_name.pt`
- Internet is required for downloading pre-trained BERT and ViT if not cached
- The app automatically uses the hybrid fusion model for prediction

---

## 📦 To Zip for Submission

To create a zip with the entire folder:

```bash
zip -r multimodal-misinformation.zip . -x "*.ipynb_checkpoints*" "__pycache__/*" ".DS_Store"
```

---

## 🙌 Acknowledgements

- 🧷 HuggingFace Transformers
- 🧠 BERT, ViT pre-trained models
- 🔥 PyTorch + Streamlit
- 🎓 This project is submitted as part of an MTech Dissertation
