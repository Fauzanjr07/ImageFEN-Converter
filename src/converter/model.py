from typing import List
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int) -> nn.Module:
    # Use ResNet18 backbone with ImageNet weights
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str = "cpu") -> nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"]) 
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
