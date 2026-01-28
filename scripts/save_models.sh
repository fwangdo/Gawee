#!/usr/bin/env bash
set -e

DIR="./torchdata"

# 1. torchdata 디렉토리 생성 (없는 경우만)
if [ ! -d "$DIR" ]; then
  mkdir "$DIR"
fi

# 2. 기존 모델 파일 삭제 (있는 경우만)
[ -f "$DIR/unet.pt" ] && rm "$DIR/unet.pt"
[ -f "$DIR/resnet18.pt" ] && rm "$DIR/resnet18.pt"

# 3. 모델 저장
python3 ./scripts/save_unet.py --path="$DIR/unet.pt"
python3 ./scripts/save_resnet.py --path="$DIR/resnet18.pt"