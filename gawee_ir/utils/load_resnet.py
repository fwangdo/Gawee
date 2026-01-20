import torch
from torchvision.models import resnet18

def load_resnet18(path: str = "./torchdata/resnet18.pt") -> torch.nn.Module:
    model = resnet18(weights=None)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model