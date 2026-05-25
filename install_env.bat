@echo off
REM =====================================================================
REM install_env.bat — Windows向け GPU加速ビジョン推論環境 自動構築スクリプト
REM 対象OS: Windows 10/11 (64-bit)
REM 前提条件: NVIDIA ドライバ >= 560, Anaconda/Miniconda インストール済み
REM =====================================================================
setlocal EnableDelayedExpansion

SET CONDA_ENV_NAME=yolo_mp_gpu
SET PYTHON_VERSION=3.12.9
SET CUDA_INDEX=cu130
SET PYTORCH_INDEX_URL=https://download.pytorch.org/whl/%CUDA_INDEX%
SET TORCH_VERSION=2.12.0+%CUDA_INDEX%
SET TORCHVISION_VERSION=0.27.0+%CUDA_INDEX%

echo.
echo =================================================================
echo   NVIDIA GPU加速 物体検出/骨格検知 環境構築スクリプト
echo   対象CUDA: %CUDA_INDEX%
echo =================================================================
echo.

REM =====================================================================
REM Step 0: NVIDIA GPU の確認
REM =====================================================================
echo [INFO] NVIDIA GPU を確認中...
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] nvidia-smi が見つかりません。GPUドライバを確認してください。
    echo [WARN] CPU動作モードで続行します。
) ELSE (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo [OK]   NVIDIA GPU を確認しました。
)

REM =====================================================================
REM Step 1: Conda の確認
REM =====================================================================
echo.
echo [INFO] Conda 環境を確認中...
conda --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda が見つかりません。
    echo         Anaconda または Miniconda をインストールしてから再実行してください。
    echo         ダウンロード先: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
FOR /F "tokens=*" %%i IN ('conda --version') DO echo [OK]   %%i が検出されました。

REM =====================================================================
REM Step 2: 既存環境の確認と削除
REM =====================================================================
echo.
conda env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [WARN] 既存の Conda 環境 '%CONDA_ENV_NAME%' が見つかりました。
    set /p REBUILD="削除して再構築しますか？ [y/N]: "
    IF /I "!REBUILD!"=="y" (
        echo [INFO] 既存環境を削除中...
        conda deactivate 2>nul
        conda env remove -n %CONDA_ENV_NAME% -y
        echo [OK]   既存環境を削除しました。
    ) ELSE (
        echo [INFO] 既存環境を保持して続行します。
        CALL conda activate %CONDA_ENV_NAME%
        echo [OK]   環境 '%CONDA_ENV_NAME%' を有効化しました。
        goto :VERIFY
    )
)

REM =====================================================================
REM Step 3: Python 3.12 環境の作成
REM =====================================================================
echo.
echo [INFO] Conda 環境 '%CONDA_ENV_NAME%' を作成中 (Python %PYTHON_VERSION%)...
conda create -n %CONDA_ENV_NAME% python=%PYTHON_VERSION% pip -y
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 環境の作成に失敗しました。
    pause
    exit /b 1
)
echo [OK]   環境を作成しました。

CALL conda activate %CONDA_ENV_NAME%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 環境のアクティベートに失敗しました。
    echo         Anaconda Prompt から再実行してください。
    pause
    exit /b 1
)
echo [OK]   環境 '%CONDA_ENV_NAME%' を有効化しました。

REM =====================================================================
REM Step 4: pip のアップグレード
REM =====================================================================
echo.
echo [INFO] pip をアップグレード中...
pip install --upgrade pip setuptools wheel
IF %ERRORLEVEL% NEQ 0 ( echo [WARN] pip アップグレードに問題が発生しました。続行します。)
echo [OK]   pip アップグレード完了。

REM =====================================================================
REM Step 5: numpy & protobuf を最優先でインストール (競合防止)
REM numpy 2.x の侵入を防ぐため、最初に個別インストールする
REM =====================================================================
echo.
echo [INFO] 基盤ライブラリ (numpy, protobuf) を優先インストール中...
pip install "numpy==1.26.4" "protobuf==4.25.6"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 基盤ライブラリのインストールに失敗しました。
    pause
    exit /b 1
)
echo [OK]   基盤ライブラリのインストール完了。

REM =====================================================================
REM Step 6: PyTorch (CUDAバイナリ) のインストール
REM =====================================================================
echo.
echo [INFO] PyTorch GPU バイナリをインストール中 (%CUDA_INDEX%)...
pip install ^
    "torch==2.7.0+%CUDA_INDEX%" ^
    "torchvision==0.22.0+%CUDA_INDEX%" ^
    --extra-index-url %PYTORCH_INDEX_URL%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch GPU バイナリのインストールに失敗しました。
    echo         CUDA インデックス URL を確認してください: %PYTORCH_INDEX_URL%
    pause
    exit /b 1
)
echo [OK]   PyTorch GPU バイナリのインストール完了。

REM =====================================================================
REM Step 7: ビジョン・推論フレームワークのインストール
REM opencv-python との共存を避けるため、contrib のみをインストール
REM =====================================================================
echo.
echo [INFO] ビジョン・推論フレームワークをインストール中...
pip install ^
    "opencv-contrib-python==4.11.0.86" ^
    "mediapipe==0.10.21" ^
    "ultralytics==8.3.100"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] フレームワークのインストールに失敗しました。
    pause
    exit /b 1
)
echo [OK]   全フレームワークのインストール完了。

REM =====================================================================
REM Step 8: 環境検証
REM =====================================================================
:VERIFY
echo.
echo [INFO] インストール環境を検証中...
python -c ^
    "import sys; print(f'  Python        : {sys.version}');^
     import numpy as np; print(f'  NumPy         : {np.__version__}');^
     assert np.__version__.startswith('1.'), f'[FAIL] NumPy 2.x が侵入: {np.__version__}';^
     import torch; print(f'  PyTorch       : {torch.__version__}');^
     print(f'  CUDA available: {torch.cuda.is_available()}');^
     cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A';^
     print(f'  GPU           : {cuda_name}');^
     import cv2; print(f'  OpenCV        : {cv2.__version__}');^
     import mediapipe; print(f'  MediaPipe     : {mediapipe.__version__}');^
     import ultralytics; print(f'  Ultralytics   : {ultralytics.__version__}');^
     print(); print('[ALL CHECKS PASSED] 環境は正常です。')"

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 環境検証に失敗しました。インストールログを確認してください。
    pause
    exit /b 1
)

echo.
echo =================================================================
echo   セットアップ完了！以下のコマンドで実行してください:
echo.
echo   conda activate %CONDA_ENV_NAME%
echo   python main.py
echo =================================================================
echo.
pause
endlocal
