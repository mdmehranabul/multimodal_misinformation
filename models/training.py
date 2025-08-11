import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import os

def train_model(model, loader, device, epochs=10, optimizer=None, start_epoch=0, checkpoint_path=None, val_loader=None):
    """
    Trains a PyTorch model with optional checkpointing.
    If checkpoint_path exists, training resumes from the saved epoch.
    
    Args:
        model: PyTorch model to train
        loader: DataLoader for training data
        device: torch.device
        epochs: total number of epochs to run
        optimizer: optional torch.optim object (creates Adam if None)
        start_epoch: starting epoch (auto-set if checkpoint loaded)
        checkpoint_path: path to save checkpoints (resumes if exists)
        val_loader: optional DataLoader for validation set to track best model
    """
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0  # Track best validation accuracy
    
    # Resume if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch} with best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for text, image, label in loader:
            text, image, label = text.to(device), image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(text, image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if val_loader:
            val_acc, val_f1 = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {val_acc:.4f}, Validation F1 Score: {val_f1:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("🔥 New best model found!")

        # Save checkpoint every epoch
        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    return model


def evaluate_model(model, loader, device):
    """
    Evaluates the model on given data loader.
    Returns accuracy and weighted F1 score.
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for text, image, label in loader:
            text, image = text.to(device), image.to(device)
            outputs = model(text, image)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return acc, f1
