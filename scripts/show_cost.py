from gawee_ir.parser import TorchParser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.analysis.cost import CostModel
from gawee_ir.passes.passer import Passer

import torch
import torch.fx as fx
import argparse

from torchvision.models import resnet18

# UNet (FX-friendly pretrained)
import segmentation_models_pytorch as smp


def load_model(model_name: str, weight_path: str | None):
    print(f'model_name -> {model_name}, weight_path -> {weight_path}')
    if model_name == "resnet18":
        model = resnet18(weights=None)
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state_dict)
        example_input = (torch.randn(1, 3, 224, 224),)

    elif model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state_dict)
        example_input = (torch.randn(1, 3, 224, 224),)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model, example_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "unet"],
        help="Model type to run (resnet18 | unet)",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="Path to model state_dict (.pt)",
    )
    parser.add_argument("--check_mode", default=False)

    args = parser.parse_args()

    # 1) Load model
    model, example_input = load_model(args.model, args.weight)

    # 2) FX trace
    gm = fx.symbolic_trace(model)

    # 3) Parse into Gawee IR
    g = TorchParser.parse_fx(gm, example_input)

    # 4) Shape inference
    ShapeInference.run(g)

    # check ir mode. 
    if args.check_mode:
        g.show_node()
        return 

    # 5) Cost before optimization
    print("== Before ==")
    CostModel.init(gm)
    CostModel.print_report(g)

    # 6) Graph rewrite passes
    Passer.run(g)

    # 7) Cost after optimization
    print("\n\n== After ==")
    CostModel.print_report(g)

    print('\n\n== Optimization information ==')
    Passer.show_opt_result()


if __name__ == "__main__":
    main()
