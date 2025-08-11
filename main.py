import sys
sys.path.append(".")

from utils.embedding_utils import get_text_embedding, get_image_embedding
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pickle
import math
import random
import numpy as np

from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
from torchvision import transforms

from models.fusion_models import EarlyFusionClassifier, LateFusionClassifier, HybridFusionClassifier
from models.dataset import MultimodalDataset
from models.training import train_model, evaluate_model

from utils.preprocessing import load_and_clean_data, download_image

# -------------------
# Seed everything for reproducibility
# -------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# -------------------
# Paths & Config
# -------------------
TRAIN_PATH = "data/multimodal_train.tsv"
VALID_PATH = "data/multimodal_validate.tsv"
TEST_PATH  = "data/multimodal_test_public.tsv"
IMAGE_DIR = "images/"
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

tqdm.pandas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Helper functions for preprocessing checkpointing
# -------------------
def save_df_checkpoint(df, filename):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)

def load_df_checkpoint(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {filename}: {e}")
            return None
    return None

def process_split_with_checkpoint_skip_done(df, split_name, checkpoint_file, batch_size=1000):
    import math
    from tqdm import tqdm

    # Initialize embedding columns if they don't exist yet
    if 'text_embedding' not in df.columns:
        df['text_embedding'] = None
    if 'image_embedding' not in df.columns:
        df['image_embedding'] = None
    if 'image_path' not in df.columns:
        df['image_path'] = None

    total_rows = len(df)
    num_batches = math.ceil(total_rows / batch_size)

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {split_name} batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()

        # Check if all embeddings exist, if yes skip
        all_text_emb = batch_df['text_embedding'].apply(lambda x: x is not None).all()
        all_img_emb = batch_df['image_embedding'].apply(lambda x: x is not None).all()
        if all_text_emb and all_img_emb:
            print(f"Skipping {split_name} rows {start_idx} to {end_idx} - embeddings already present.")
            continue

        print(f"Processing {split_name} rows {start_idx} to {end_idx}...")

        # Download images if image_path missing
        missing_img_path_mask = batch_df['image_path'].isnull() | (batch_df['image_path'] == '')
        if missing_img_path_mask.any():
            batch_df.loc[missing_img_path_mask, 'image_path'] = batch_df.loc[missing_img_path_mask].progress_apply(
                lambda row: download_image(row, IMAGE_DIR), axis=1)

        # Generate missing text embeddings
        missing_text_emb_mask = batch_df['text_embedding'].apply(lambda x: x is None)
        if missing_text_emb_mask.any():
            batch_df.loc[missing_text_emb_mask, 'text_embedding'] = batch_df.loc[missing_text_emb_mask, 'title'].progress_apply(
                lambda x: get_text_embedding(x, bert_tokenizer, bert_model, device))

        # Generate missing image embeddings
        missing_img_emb_mask = batch_df['image_embedding'].apply(lambda x: x is None)
        if missing_img_emb_mask.any():
            batch_df.loc[missing_img_emb_mask, 'image_embedding'] = batch_df.loc[missing_img_emb_mask, 'image_path'].progress_apply(
                lambda x: get_image_embedding(x, image_transform, vit_model, device))

        # Drop rows with failed embedding generation (None)
        batch_df.dropna(subset=['text_embedding', 'image_embedding'], inplace=True)

        # Update original dataframe rows with batch results
        df.loc[batch_df.index, ['text_embedding', 'image_embedding', 'image_path']] = batch_df.loc[:, ['text_embedding', 'image_embedding', 'image_path']]

        # Save checkpoint after processing this batch
        print(f"Saving checkpoint for {split_name} rows {start_idx}-{end_idx}...")
        save_df_checkpoint(df, checkpoint_file)
        print(f"Checkpoint saved.")

    return df

# -------------------
# Load pretrained models
# -------------------
print("Loading BERT and VIT models...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=vit_extractor.image_mean, std=vit_extractor.image_std),
])

# -------------------
# Load datasets
# -------------------
print("Loading and cleaning data...")
train_df = load_and_clean_data(TRAIN_PATH, sample_size=200, seed=42)
valid_df = load_and_clean_data(VALID_PATH, sample_size=200, seed=42)
test_df  = load_and_clean_data(TEST_PATH,  sample_size=200, seed=42)

def filter_missing_text_or_image(df):
    return df[(df['title'].notnull()) & (df['title'] != '') & (df['image_url'].notnull()) & (df['image_url'] != '')].reset_index(drop=True)

train_df = filter_missing_text_or_image(train_df)
valid_df = filter_missing_text_or_image(valid_df)
test_df  = filter_missing_text_or_image(test_df)

# -------------------
# Preprocessing with checkpointing + skipping done batches
# -------------------
for df, split_name in zip([train_df, valid_df, test_df], ['train', 'valid', 'test']):
    checkpoint_file = os.path.join(MODEL_SAVE_DIR, f"{split_name}_processed.pkl")

    cached_df = load_df_checkpoint(checkpoint_file)
    if cached_df is not None:
        print(f"Loaded {split_name} split from checkpoint.")
        if split_name == 'train':
            train_df = cached_df
        elif split_name == 'valid':
            valid_df = cached_df
        else:
            test_df = cached_df
        continue

    print(f"\nProcessing {split_name} split with checkpointing and skipping done rows...")
    processed_df = process_split_with_checkpoint_skip_done(df, split_name, checkpoint_file, batch_size=1000)
    if split_name == 'train':
        train_df = processed_df
    elif split_name == 'valid':
        valid_df = processed_df
    else:
        test_df = processed_df

# After all splits processed or loaded from checkpoint
train_df = train_df.dropna(subset=['text_embedding', 'image_embedding']).reset_index(drop=True)
valid_df = valid_df.dropna(subset=['text_embedding', 'image_embedding']).reset_index(drop=True)
test_df = test_df.dropna(subset=['text_embedding', 'image_embedding']).reset_index(drop=True)


# -------------------
# Plot distributions
# -------------------
from utils.analysis import plot_train_label_distributions
plot_train_label_distributions(train_df)

# -------------------
# Training & Evaluation
# -------------------
for label_type in ['2_way_label_name', '6_way_label_name']:
    print(f"\n========== {label_type.upper()} CLASSIFICATION ==========")

    # Encode labels
    le = LabelEncoder()
    combined_labels = pd.concat([train_df[label_type], valid_df[label_type], test_df[label_type]])
    le.fit(combined_labels)

    train_df['label'] = le.transform(train_df[label_type])
    valid_df['label'] = le.transform(valid_df[label_type])
    test_df['label']  = le.transform(test_df[label_type])

    label_map = dict(zip(range(len(le.classes_)), le.classes_))
    print("Label Mapping:", label_map)

    # Create datasets
    train_dataset = MultimodalDataset(train_df)
    valid_dataset = MultimodalDataset(valid_df)
    test_dataset  = MultimodalDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader  = DataLoader(test_dataset, batch_size=32)

    output_dim = len(le.classes_)
    fusion_results = {}

    # -------------------
    # Loop through fusion models
    # -------------------
    for name, ModelClass in zip(
        ['Early Fusion', 'Late Fusion', 'Hybrid Fusion'],
        [EarlyFusionClassifier, LateFusionClassifier, HybridFusionClassifier]
    ):
        print(f"\nTraining {name} model...")
        model = ModelClass(output_dim=output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        start_epoch = 0

        # Load training checkpoint if exists
        checkpoint_file = os.path.join(MODEL_SAVE_DIR, f"{name.replace(' ', '_').lower()}_{label_type}_checkpoint.pth")
        if os.path.exists(checkpoint_file):
            print(f"Resuming {name} ({label_type}) from checkpoint...")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            print(f"Resumed from epoch {start_epoch}")

        # Train with checkpoint saving every epoch
        trained_model = train_model(
            model,
            train_loader,
            device=device,
            epochs=10,
            optimizer=optimizer,
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_file,
            val_loader=valid_loader  # To track best_val_acc during training
        )

        val_acc, val_f1 = evaluate_model(trained_model, valid_loader, device=device)
        test_acc, test_f1 = evaluate_model(trained_model, test_loader, device=device)

        fusion_results[name] = {
            'val_accuracy': val_acc, 'val_f1_score': val_f1,
            'test_accuracy': test_acc, 'test_f1_score': test_f1
        }

        # Save final hybrid model for deployment
        if name == 'Hybrid Fusion':
            final_model_path = os.path.join(MODEL_SAVE_DIR, f"hybrid_fusion_{label_type}.pt")
            torch.save(trained_model.state_dict(), final_model_path)
            print(f"Model saved to: {final_model_path}")

    # Print comparison
    print(f"\n--- Fusion Strategy Comparison ({label_type}) ---")
    for strategy, metrics in fusion_results.items():
        print(f"{strategy}:")
        print(f"  Val Accuracy = {metrics['val_accuracy']:.4f}, Val F1 Score = {metrics['val_f1_score']:.4f}")
        print(f"  Test Accuracy = {metrics['test_accuracy']:.4f}, Test F1 Score = {metrics['test_f1_score']:.4f}")
