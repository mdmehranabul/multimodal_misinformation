# 📚 Multimodal Misinformation Detection

This project is a **dissertation-level solution** for detecting **misinformation in social media posts** using both **text and image modalities**. It leverages **deep learning** with **BERT** and **Vision Transformer (ViT)** models, implements multiple **fusion strategies**, and supports both **2-way** and **6-way** classification. Additionally, it integrates **Gemini multimodal outputs** for benchmarking.

---

## 🚀 Features

* **Encoders:**

  * 📝 Text → BERT (`bert-base-uncased`)
  * 🖼️ Image → Vision Transformer (`vit-base-patch16-224`)
* **Fusion Models:**

  * 🔗 Early Fusion
  * 🔗 Late Fusion
  * 🔗 Hybrid Fusion
* **Classification Tasks:**

  * ✅ **2-Way:** Real vs Misleading
  * ✅ **6-Way:** Real, Satire, Misleading, Manipulated, Fake Connection, Imposter Content
* **Extras:**

  * Unimodal baselines (`train_unimodal.py`)
  * Embedding visualization (`view_embeddings.py`)
  * Gemini multimodal benchmarking (`gemini_classifier.py`)
  * Exploratory data analysis & statistical insights
  * Fully functional **Streamlit app** for end-user predictions

---

## 🗂️ Project Structure

```
.
├── data/                         # Dataset files
├── images/                       # Supporting images/plots
├── models/                       # Model architectures
│   ├── dataset.py                # Dataset class
│   ├── fusion_models.py          # Early, Late, Hybrid fusion models
│   └── training.py               # Training/evaluation utilities
├── plots/                        # Output plots
├── saved_models/                 # Pre-trained & best checkpoints
├── utils/                        # Preprocessing, embedding, and analysis scripts
│
├── analysis_stats.py              # Statistical analysis
├── calc.py                        # Helper calculations
├── check_checkpoints.py           # Debugging checkpoints
├── check_missing_raw_fields.py    # Data validation
├── check_train_embeddings.py      # Verify embedding generation
├── config.py                      # Configurations
├── gemini_classifier.py           # Gemini multimodal baseline
├── gemini_output_2_way.csv        # Gemini 2-way predictions
├── gemini_output_6_way.csv        # Gemini 6-way predictions
├── main.py                        # Train & evaluate fusion models
├── streamlit_app.py               # Streamlit web interface
├── train_unimodal.py              # Train text-only & image-only baselines
├── view_embeddings.py              # Embedding visualization
│
├── 2_way_label_distribution.png   # Label distribution plots
├── 6_way_label_distribution.png
├── post_year_distribution.png
├── requirements.txt
└── README.md
```

---

## 📊 Exploratory Analysis

* Distribution plots: `2_way_label_distribution.png`, `6_way_label_distribution.png`
* Temporal trends: `post_year_distribution.png`
* Scripts: `analysis_stats.py`, `utils/analysis.py`

---

## 🗃️ Dataset

* **File:** `data/multimodal_test_public.tsv`
* **Columns:** `title`, `image_url`, `2_way_label`, `6_way_label`, `created_utc`

---

## 🧠 Training & Evaluation

Train and evaluate across both 2-way and 6-way tasks:

```bash
python main.py
```

Train unimodal baselines:

```bash
python train_unimodal.py
```

Evaluate Gemini multimodal baseline:

```bash
python gemini_classifier.py
```

Results include accuracy, F1-score, and saved best models in `saved_models/`.

---

## 🌐 Streamlit App

Launch the app for predictions:

```bash
streamlit run streamlit_app.py
```

Features:

* Upload image and enter caption
* Select between 2-way or 6-way classification
* Output: predicted label + confidence visualization

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:

* `transformers`
* `torch`
* `streamlit`
* `scikit-learn`
* `pandas`, `matplotlib`, `seaborn`

---

## 📌 Notes

* Ensure `saved_models/` contains:

  * `hybrid_fusion_2_way_label_name.pt`
  * `hybrid_fusion_6_way_label_name.pt`
* Internet required for downloading pretrained BERT/ViT
* Gemini outputs (`gemini_output_*.csv`) serve as benchmarks

---

## 📦 Packaging for Submission

To create a zip for submission:

```bash
zip -r multimodal-misinformation.zip . -x "*.ipynb_checkpoints*" "__pycache__/*" ".DS_Store"
```

---

## 📐 Architecture Diagram

```text
            +-------------------+         +-------------------+
            |    Text Encoder   |         |   Image Encoder   |
            |       (BERT)      |         |        (ViT)      |
            +-------------------+         +-------------------+
                       |                           |
                       |                           |
                       +-----------+---------------+
                                   |
                         [ Fusion Layer ]
                (Early Fusion | Late Fusion | Hybrid)
                                   |
                          +-----------------+
                          |   Classifier    |
                          +-----------------+
                                   |
                        Prediction (2-way / 6-way)
```

---

## 🙌 Acknowledgements

* HuggingFace Transformers (BERT, ViT)
* PyTorch
* Streamlit
* Google Gemini (benchmarking)
* M.Tech Dissertation Guidance