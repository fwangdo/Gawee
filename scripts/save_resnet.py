import torch
from torchvision.models import resnet18, ResNet18_Weights
import argparse

def save_resnet18():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--path", type=str, help="Path to save the ResNet-18 weights")
    args = parser.parse_args()
    path = args.path    
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()  # inference 기준
    torch.save(model.state_dict(), path)
    print(f"Saved ResNet-18 weights to {path}")

if __name__ == "__main__":
    save_resnet18()