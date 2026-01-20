from gawee_ir.parser import TorchParser
import argparse
import torch    
from torchvision.models import resnet18
import torch.fx as fx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    model = resnet18(weights=None)
    state_dict = torch.load(args.path)
    model.load_state_dict(state_dict)
    model.eval()

    gm = fx.symbolic_trace(model)
    g = TorchParser.parse_fx(gm, (torch.randn(1, 3, 224, 224),))    

    print(f'\n\n\nIR representation.')
    g.dump()
    return 


if __name__ == "__main__":
    main()