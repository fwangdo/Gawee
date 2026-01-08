# models/resnet18.py
import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 1000) -> nn.Module:
    """
    가장 표준적인 ResNet-18.
    pretrained 사용하지 않음 (그래프 구조 분석 목적)
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    model = build_resnet18()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
