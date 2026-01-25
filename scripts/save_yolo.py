import argparse
import torch
from ultralytics import YOLO

def save_yolov8n():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to save YOLOv8n model (.pt)",
    )
    args = parser.parse_args()
    path = args.path

    # Load pretrained YOLOv8 nano (edge-friendly, most common)
    yolo = YOLO("yolov8n.pt")

    # Extract pure torch.nn.Module
    model = yolo.model
    model.eval()

    # Save full torch module (recommended for FX / compiler work)
    torch.save(model, path)

    print(f"Saved YOLOv8n torch model to {path}")


if __name__ == "__main__":
    save_yolov8n()
