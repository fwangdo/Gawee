# scripts/inspect_model.py
import torch
from models.resnet18 import build_resnet18
from collections import defaultdict


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def analyze_model(model, input_tensor):
    flops = 0
    activations = 0
    hooks = []

    def hook_fn(module, inp, out):
        nonlocal flops, activations

        # activation count
        if isinstance(out, torch.Tensor):
            activations += out.numel()

        # FLOPs (아주 보수적인 추정)
        if isinstance(module, torch.nn.Conv2d):
            out_h, out_w = out.shape[2:]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            flops += (
                out_h
                * out_w
                * module.out_channels
                * (module.in_channels / module.groups)
                * kernel_ops
            )

        elif isinstance(module, torch.nn.Linear):
            flops += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return flops, activations


def main():
    model = build_resnet18()
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    # 1. 컴파일 가능 여부 (forward 실행)
    with torch.no_grad():
        y = model(x)
    print("Forward OK, output shape:", y.shape)

    # 2. 파라미터 수
    params = count_parameters(model)
    print(f"Parameters: {params:,}")

    # 3. FLOPs / activation 분석
    flops, activations = analyze_model(model, x)
    print(f"Estimated FLOPs: {int(flops):,}")
    print(f"Activation elements: {activations:,}")

    # 4. 메모리 추정 (FP32 기준)
    param_mem = params * 4 / (1024 ** 2)
    activation_mem = activations * 4 / (1024 ** 2)

    print(f"Parameter memory (MB): {param_mem:.2f}")
    print(f"Activation memory (MB): {activation_mem:.2f}")


if __name__ == "__main__":
    main()
