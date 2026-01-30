from __future__ import annotations
from typing     import *

from gawee_ir.parser         import TorchParser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.passes.passer  import Passer
from gawee_ir.translator     import Translator

import torch
import torch.fx as fx
import argparse

from torchvision.models import resnet18

# UNet (FX-friendly pretrained)
import segmentation_models_pytorch as smp


# TODO: wee need to make loader. 
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


def main() -> None:
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
    args = parser.parse_args()

    # 1) Load model
    model, example_input = load_model(args.model, args.weight)

    # 2) FX trace
    gm = fx.symbolic_trace(model)

    # 3) Parse into Gawee IR
    g = TorchParser.parse_fx(gm, example_input)

    # 4) Shape inference
    ShapeInference.run(g)

    # 5) Graph rewrite passes
    Passer.run(g)

    # 6) Translate graph into json and bin. 
    path = "jsondata"
    trans = Translator(path)
    trans.export(g)
    return 


if __name__ == "__main__":
    main()
