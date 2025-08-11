# models/fusion_models.py

import torch
import torch.nn as nn

class EarlyFusionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6):
        super(EarlyFusionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_feat, image_feat):
        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.fc(combined)


class LateFusionClassifier(nn.Module):
    def __init__(self, input_dim=768, output_dim=6):
        super(LateFusionClassifier, self).__init__()
        self.text_fc = nn.Linear(input_dim, output_dim)
        self.image_fc = nn.Linear(input_dim, output_dim)

    def forward(self, text_feat, image_feat):
        text_logits = self.text_fc(text_feat)
        image_logits = self.image_fc(image_feat)
        return (text_logits + image_logits) / 2


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