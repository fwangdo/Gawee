import torch
import argparse
import segmentation_models_pytorch as smp


def save_unet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save UNet weights")
    args = parser.parse_args()

    model = smp.Unet(
        encoder_name="resnet34",        # 매우 흔한 조합
        encoder_weights="imagenet",     # pretrained
        in_channels=3,
        classes=1,
    )
    model.eval()  # inference 기준

    torch.save(model.state_dict(), args.path)
    print(f"Saved UNet weights to {args.path}")


if __name__ == "__main__":
    save_unet()
