import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FlowerMultiOutputModel(nn.Module):
    def __init__(self, num_flower_types=5, num_colors=4):
        super().__init__()

        # Load pretrained ResNet50 backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the classification head (fc layer)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        # Define three separate heads for multi-task learning
        self.flower_head = nn.Linear(in_features, num_flower_types)
        self.color_head = nn.Linear(in_features, num_colors)
        self.oil_head = nn.Linear(in_features, 3)  # For 3 oil concentration regressions

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten

        flower_logits = self.flower_head(x)
        color_logits = self.color_head(x)
        oil_preds = self.oil_head(x)

        return flower_logits, color_logits, oil_preds
