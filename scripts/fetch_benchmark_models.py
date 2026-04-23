from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


NLP_BENCHMARKS = {
    "bert_tiny": {
        "repo_id": "onnx-community/bert-tiny-finetuned-sms-spam-detection-ONNX",
        "revision": "358e80a313103279be7292e32d112091c91de10b",
    },
    "distilbert_base_uncased": {
        "repo_id": "onnx-community/distilbert-base-uncased-ONNX",
        "revision": "a5d2f36",
    },
}


def fetch_model(name: str, repo_id: str, revision: str, target_root: Path) -> None:
    local_dir = target_root / name
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data"],
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[OK] Downloaded {name} from {repo_id}@{revision} to {local_dir}")


def main() -> None:
    target_root = Path("benchmarks/onnx/nlp")
    target_root.mkdir(parents=True, exist_ok=True)
    for name, spec in NLP_BENCHMARKS.items():
        fetch_model(name, spec["repo_id"], spec["revision"], target_root)


if __name__ == "__main__":
    main()
