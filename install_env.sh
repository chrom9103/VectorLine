#!/usr/bin/env bash
# =====================================================================
# install_env.sh — Linux向け GPU加速ビジョン推論環境 自動構築スクリプト
# 対象OS: Ubuntu 22.04 LTS / 24.04 LTS
# 前提条件: NVIDIA ドライバ ≥ 560, CUDA 13.0 または 12.6 インストール済み
# =====================================================================
set -euo pipefail

CONDA_ENV_NAME="yolo_mp_gpu"
PYTHON_VERSION="3.12.9"
CUDA_INDEX="cu130"  # CUDA 12.6の場合は cu126 に変更
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_INDEX}"

# --- カラー出力の定義 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET} $1"; }
log_success() { echo -e "${GREEN}[OK]${RESET}   $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }

# =====================================================================
# Step 0: CUDA ドライバの確認
# =====================================================================
log_info "NVIDIA GPU / CUDA ドライバを確認中..."
if ! command -v nvidia-smi &> /dev/null; then
    log_warn "nvidia-smi が見つかりません。GPUドライバのインストールを確認してください。"
    log_warn "CPU動作モードで続行します。"
else
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    log_success "NVIDIA GPU を確認しました。"
fi

# =====================================================================
# Step 1: Conda の確認とインストール
# =====================================================================
log_info "Conda 環境を確認中..."
if ! command -v conda &> /dev/null; then
    log_warn "Conda が見つかりません。Miniconda をインストールします..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    wget -q --show-progress -O "${MINICONDA_INSTALLER}" "${MINICONDA_URL}"
    bash "${MINICONDA_INSTALLER}" -b -p "$HOME/miniconda3"
    rm -f "${MINICONDA_INSTALLER}"
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    log_success "Miniconda のインストールが完了しました。"
else
    log_success "Conda が検出されました: $(conda --version)"
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# =====================================================================
# Step 2: 既存環境の削除（再構築オプション）
# =====================================================================
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log_warn "既存の Conda 環境 '${CONDA_ENV_NAME}' が見つかりました。"
    read -rp "削除して再構築しますか？ [y/N]: " REBUILD
    if [[ "${REBUILD:-N}" =~ ^[Yy]$ ]]; then
        conda deactivate 2>/dev/null || true
        conda env remove -n "${CONDA_ENV_NAME}" -y
        log_success "既存環境を削除しました。"
    else
        log_info "既存環境を保持して続行します。"
        conda activate "${CONDA_ENV_NAME}"
        log_success "環境 '${CONDA_ENV_NAME}' を有効化しました。"
        exit 0
    fi
fi

# =====================================================================
# Step 3: Python 3.12 環境の作成
# =====================================================================
log_info "Conda 環境 '${CONDA_ENV_NAME}' を作成中 (Python ${PYTHON_VERSION})..."
conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" pip -y
conda activate "${CONDA_ENV_NAME}"
log_success "環境 '${CONDA_ENV_NAME}' を有効化しました。"

# =====================================================================
# Step 4: pip のアップグレード
# =====================================================================
log_info "pip をアップグレード中..."
pip install --upgrade pip setuptools wheel
log_success "pip のアップグレード完了。"

# =====================================================================
# Step 5: numpy & protobuf を最優先でインストール (競合防止)
# numpy 2.x の侵入を防ぐため、最初に個別インストールする
# =====================================================================
log_info "基盤ライブラリ (numpy, protobuf) を優先インストール中..."
pip install "numpy==1.26.4" "protobuf==4.25.6"
log_success "基盤ライブラリのインストール完了。"

# =====================================================================
# Step 6: PyTorch (CUDAバイナリ) のインストール
# =====================================================================
log_info "PyTorch GPU バイナリをインストール中 (${CUDA_INDEX})..."
pip install \
    "torch==2.7.0+${CUDA_INDEX}" \
    "torchvision==0.22.0+${CUDA_INDEX}" \
    --extra-index-url "${PYTORCH_INDEX_URL}"
log_success "PyTorch GPU バイナリのインストール完了。"

# =====================================================================
# Step 7: ビジョン・推論フレームワークのインストール
# opencv-python との共存を避けるため、contrib のみをインストール
# =====================================================================
log_info "ビジョン・推論フレームワークをインストール中..."
pip install \
    "opencv-contrib-python==4.11.0.86" \
    "mediapipe==0.10.21" \
    "ultralytics==8.3.100"
log_success "全フレームワークのインストール完了。"

# =====================================================================
# Step 8: 環境検証
# =====================================================================
log_info "インストール環境を検証中..."
python - <<'PYEOF'
import sys
print(f"  Python バージョン : {sys.version}")

import numpy as np
print(f"  NumPy バージョン  : {np.__version__}")
assert np.__version__.startswith("1."), f"[FAIL] NumPy 2.x が侵入しています: {np.__version__}"

import torch
print(f"  PyTorch バージョン: {torch.__version__}")
print(f"  CUDA 利用可能     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU デバイス名    : {torch.cuda.get_device_name(0)}")

import cv2
print(f"  OpenCV バージョン : {cv2.__version__}")

import mediapipe
print(f"  MediaPipe バージョン: {mediapipe.__version__}")

import ultralytics
print(f"  Ultralytics バージョン: {ultralytics.__version__}")

print("\n[ALL CHECKS PASSED] 環境は正常です。")
PYEOF

log_success "環境検証が完了しました。"

echo ""
echo -e "${GREEN}=================================================================${RESET}"
echo -e "${GREEN}  セットアップ完了！以下のコマンドで環境を有効化してください:${RESET}"
echo -e "${GREEN}  $ conda activate ${CONDA_ENV_NAME}${RESET}"
echo -e "${GREEN}  $ python main.py${RESET}"
echo -e "${GREEN}=================================================================${RESET}"
