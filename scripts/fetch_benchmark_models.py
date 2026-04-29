from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import onnx
from huggingface_hub import hf_hub_download, snapshot_download
from ultralytics import YOLO


HUB_BENCHMARKS = {
    "bert_tiny": {
        "repo_id": "onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX",
        "revision": "358e80a313103279be7292e32d112091c91de10b",
        "target_dir": Path("benchmarks/onnx/nlp/bert_tiny"),
        "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data"],
    },
    "distilbert_base_uncased": {
        "repo_id": "onnx-community/distilbert-base-uncased-ONNX",
        "revision": "a5d2f36",
        "target_dir": Path("benchmarks/onnx/nlp/distilbert_base_uncased"),
        "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data"],
    },
    "distilbert_base_uncased_mnli": {
        "repo_id": "onnx-community/distilbert-base-uncased-mnli-ONNX",
        "revision": "19a5e67",
        "target_dir": Path("benchmarks/onnx/nlp/distilbert_base_uncased_mnli"),
        "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data"],
    },
    "vit_tiny_patch16_224": {
        "repo_id": "onnx-community/vit-tiny-patch16-224-ONNX",
        "revision": "ebffd7b9c0c53b51bada44116abd8f01aed41be6",
        "target_dir": Path("benchmarks/onnx/vision/vit_tiny_patch16_224"),
        "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data"],
    },
    "vit_base_patch16_224": {
        "repo_id": "onnx-community/vit-base-patch16-224-ONNX",
        "revision": "cbd8c61cf91321b8dc986b5cbf84af9db7eac1a7",
        "target_dir": Path("benchmarks/onnx/vision/vit_base_patch16_224"),
        "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data"],
    },
    "dinov3_convnext_tiny": {
        "repo_id": "onnx-community/dinov3-convnext-tiny-pretrain-lvd1689m-ONNX",
        "revision": "4294cd5539afe2380d859e21e1e7925c127f2dfc",
        "target_dir": Path("benchmarks/onnx/vision/dinov3_convnext_tiny"),
        "allow_patterns": ["onnx/model.onnx", "onnx/model.onnx_data"],
    },
    "qwen3_0_6b": {
        "repo_id": "broadfield-dev/Qwen3-0.6B-onnx",
        "revision": None,
        "target_dir": Path("benchmarks/onnx/nlp/qwen3_0_6b"),
        "allow_patterns": ["model.onnx", "model.onnx_data", "config.json", "tokenizer.json"],
        "expected_model": Path("model.onnx"),
    },
}

PT_EXPORT_BENCHMARKS = {
    "yolo26_n": {
        "repo_id": "openvision/yolo26-n",
        "revision": None,
        "target_dir": Path("benchmarks/onnx/vision/yolo26_n"),
        "pt_filename": "model.pt",
        "opset": 17,
        "imgsz": 640,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or export benchmark ONNX models.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional benchmark names to fetch. If omitted, fetch all configured models.",
    )
    return parser.parse_args()


def selected_specs(only: list[str] | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    hub_specs = HUB_BENCHMARKS
    pt_specs = PT_EXPORT_BENCHMARKS
    if not only:
        return hub_specs, pt_specs

    wanted = set(only)
    filtered_hub = {name: spec for name, spec in hub_specs.items() if name in wanted}
    filtered_pt = {name: spec for name, spec in pt_specs.items() if name in wanted}
    missing = wanted - set(filtered_hub) - set(filtered_pt)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(f"Unknown benchmark names: {missing_text}")
    return filtered_hub, filtered_pt


def fetch_model(
    name: str,
    repo_id: str,
    revision: str | None,
    target_dir: Path,
    allow_patterns: list[str],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "allow_patterns": allow_patterns,
        "local_dir": target_dir,
        "local_dir_use_symlinks": False,
    }
    if revision is not None:
        kwargs["revision"] = revision
    snapshot_download(**kwargs)
    revision_text = revision or "default-branch"
    print(f"[OK] Downloaded {name} from {repo_id}@{revision_text} to {target_dir}")


def verify_onnx_opset(name: str, model_path: Path, minimum_opset: int = 13) -> None:
    model = onnx.load(model_path, load_external_data=False)
    imported = {opset.domain or "ai.onnx": opset.version for opset in model.opset_import}
    ai_onnx_opset = imported.get("ai.onnx")
    if ai_onnx_opset is None:
        raise RuntimeError(f"{name}: ai.onnx opset is missing in {model_path}")
    if ai_onnx_opset < minimum_opset:
        raise RuntimeError(
            f"{name}: ai.onnx opset {ai_onnx_opset} is below required minimum {minimum_opset}"
        )
    print(f"[OK] {name} uses ai.onnx opset {ai_onnx_opset}")


def export_yolo_model(
    name: str,
    repo_id: str,
    revision: str | None,
    target_dir: Path,
    pt_filename: str,
    opset: int,
    imgsz: int,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": pt_filename,
        "local_dir": target_dir,
        "local_dir_use_symlinks": False,
    }
    if revision is not None:
        download_kwargs["revision"] = revision
    pt_path = Path(hf_hub_download(**download_kwargs))

    model = YOLO(str(pt_path))
    exported = Path(
        model.export(format="onnx", opset=opset, dynamic=False, simplify=False, imgsz=imgsz)
    )
    onnx_path = target_dir / "model.onnx"
    if exported.resolve() != onnx_path.resolve():
        shutil.move(str(exported), str(onnx_path))

    data_path = exported.with_suffix(".onnx_data")
    target_data_path = target_dir / "model.onnx_data"
    if data_path.exists() and data_path.resolve() != target_data_path.resolve():
        shutil.move(str(data_path), str(target_data_path))

    revision_text = revision or "default-branch"
    print(
        f"[OK] Downloaded {name} weights from {repo_id}@{revision_text} and exported ONNX to {onnx_path}"
    )


def main() -> None:
    args = parse_args()
    hub_specs, pt_specs = selected_specs(args.only)

    for name, spec in hub_specs.items():
        fetch_model(
            name,
            spec["repo_id"],
            spec["revision"],
            spec["target_dir"],
            spec["allow_patterns"],
        )
        expected_model = spec.get("expected_model")
        if expected_model is not None:
            verify_onnx_opset(name, spec["target_dir"] / expected_model)

    for name, spec in pt_specs.items():
        export_yolo_model(
            name,
            spec["repo_id"],
            spec["revision"],
            spec["target_dir"],
            spec["pt_filename"],
            spec["opset"],
            spec["imgsz"],
        )
        verify_onnx_opset(name, spec["target_dir"] / "model.onnx")


if __name__ == "__main__":
    main()
