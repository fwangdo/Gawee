import argparse
import torch
import torchvision

def main(out_path: str):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    x = torch.randn(1, 3, 224, 224)  # fixed input shape
    torch.onnx.export(
        model,
        x,
        out_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    print("exported:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out)
