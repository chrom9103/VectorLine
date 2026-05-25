"""
===========================================================================
骨格検知 (MediaPipe Pose) — 33点リアルタイム骨格トラッキングモジュール
非同期カメラキャプチャ対応 スレッドセーフ推論
===========================================================================
前提ライブラリ:
    mediapipe == 0.10.21    (MediaPipe Pose, 33-landmark)
    opencv-contrib-python == 4.11.0.86
    numpy == 1.26.4         (MediaPipe の C-API 要件: numpy < 2.x)
    protobuf == 4.25.6
===========================================================================
"""

import cv2
import sys
import time
import threading
import numpy as np
import mediapipe as mp


# ===========================================================================
# A. スレッドセーフな非同期カメラキャプチャクラス
# ===========================================================================
class ThreadedCamera:
    """
    USBカメラ (または内蔵カメラ) のフレームを、推論メインループとは独立した
    専用デーモンスレッドで連続的に取得するクラス。

    推論の処理時間に関わらず常に最新フレームをバッファリングし、
    メインループからの読み取り時には深コピー (np.copy) を返すことで、
    スレッド間データ競合 (Race Condition) を完全に防止する。
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720) -> None:
        """
        引数:
            src   : カメラデバイスインデックス (通常 0 が内蔵/デフォルトカメラ)
            width : キャプチャ解像度 横幅 (ピクセル)
            height: キャプチャ解像度 縦幅 (ピクセル)
        """
        # DirectShow バックエンドで開く
        # MSMF は NVIDIA Broadcast 等の常駐アプリと競合しフレーム取得失敗する場合がある
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # フォールバック: バックエンド指定なし (OS任せ)
            print(f"[!] DirectShow で開けませんでした。デフォルトバックエンドで再試行します...")
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[FATAL] カメラデバイス (src={src}) を開けません。"
                " USBカメラが接続されているか、デバイスインデックスを確認してください。"
            )

        # 解像度の設定 (デバイスが対応していない場合は無視されることがある)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 初期フレームの取得 (スレッド開始前のウォームアップ)
        self.grabbed, self.frame = self.cap.read()

        self.started = False
        self.read_lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def start(self) -> "ThreadedCamera":
        """キャプチャスレッドを起動する。二重起動は無視される。"""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True, name="CameraThread")
        self.thread.start()
        return self

    def _update(self) -> None:
        """[内部メソッド] バックグラウンドスレッドで連続フレームを取得し続けるループ。"""
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        最新フレームの深コピーを返す。

        戻り値:
            grabbed (bool)          : フレームの取得成功フラグ
            frame (np.ndarray|None) : BGR フォーマットのフレームコピー
        """
        with self.read_lock:
            # メインループ側でフレームを加工している間にスレッドが
            # self.frame を上書きしないよう、必ず深コピーを返す
            if self.grabbed and self.frame is not None:
                return True, np.copy(self.frame)
            return False, None

    def stop(self) -> None:
        """キャプチャスレッドを停止し、VideoCaptureリソースを解放する。"""
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.cap.release()
        print("[*] カメラリソースを解放しました。")


# ===========================================================================
# B. メイン骨格検知システム
# ===========================================================================
def main() -> None:
    """
    MediaPipe Pose による 33点リアルタイム骨格トラッキングループ。

    処理フロー:
        1. MediaPipe Pose (33ランドマーク) の初期化
        2. ThreadedCamera の起動
        3. 推論・描画・表示ループ
        4. クリーンアップ (finally ブロックで確実に実行)
    """

    # -----------------------------------------------------------------------
    # 1. MediaPipe Pose の初期化
    # -----------------------------------------------------------------------
    print("[*] MediaPipe Pose を初期化中...")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose_estimator = mp_pose.Pose(
        static_image_mode=False,       # 動画ストリーム向けトラッキングモード
        model_complexity=1,            # 0=Lite / 1=Full / 2=Heavy
        smooth_landmarks=True,         # ジッター抑制フィルタ有効化
        enable_segmentation=False,     # セグメンテーションは不使用 (負荷削減)
        smooth_segmentation=False,
        min_detection_confidence=0.5,  # 初期検出の信頼度閾値
        min_tracking_confidence=0.5,   # トラッキング維持の信頼度閾値
    )
    print("[*] MediaPipe Pose の初期化完了。")

    # -----------------------------------------------------------------------
    # 2. 非同期カメラキャプチャの起動
    # -----------------------------------------------------------------------
    print("\n[*] カメラデバイスを起動中 (src=0, 1280x720)...")
    try:
        camera = ThreadedCamera(src=0, width=1280, height=720)
        camera.start()
    except RuntimeError as e:
        print(f"[!] {e}")
        pose_estimator.close()
        sys.exit(1)

    # カメラバッファが満たされるまで待機 (最初の数フレームはゴミデータの場合がある)
    time.sleep(1.0)
    print("[*] 推論ループを開始します。ウィンドウ上で 'q' キーを押すと終了します。\n")

    # -----------------------------------------------------------------------
    # 3. 推論・描画ループ
    # -----------------------------------------------------------------------
    prev_time = time.perf_counter()
    frame_count = 0

    try:
        while True:
            # --- フレームの取得 ---
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                print("[!] フレームのデコードに失敗しました。カメラ接続を確認してください。")
                break

            frame_count += 1

            # --- (a) MediaPipe 用 BGR → RGB 変換 ---
            # MediaPipe の process() は RGB 入力を要求する。
            # OpenCV のデフォルト出力は BGR であるため、必ず変換すること。
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # writeable=False にすることで内部コピーを回避しメモリ転送を最適化
            rgb_frame.flags.writeable = False

            # --- (b) MediaPipe Pose 骨格推論 ---
            pose_results = pose_estimator.process(rgb_frame)

            # writeable フラグを戻す (描画時に必要)
            rgb_frame.flags.writeable = True

            # --- (c) 骨格ランドマークの描画 ---
            # draw_landmarks は BGR フレームへ直接描画する (内部で色変換は行わない)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # --- (d) FPS の HUD 表示 ---
            curr_time = time.perf_counter()
            elapsed = curr_time - prev_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            prev_time = curr_time

            hud_text = f"MediaPipe Pose | FPS: {fps:.1f} | Frame: {frame_count}"
            # 影付き文字で視認性を高める
            cv2.putText(frame, hud_text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(frame, hud_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)

            # --- (e) フレームの表示 ---
            cv2.imshow("VectorLine — MediaPipe Pose 33-Landmark Skeleton", frame)

            # 'q' キーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[*] 終了キー ('q') を検出しました。シャットダウンを開始します...")
                break

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C による割り込みを検出しました。シャットダウンを開始します...")

    finally:
        # -----------------------------------------------------------------------
        # 4. クリーンアップ (確実なリソース解放)
        # -----------------------------------------------------------------------
        print("[*] リソースを解放中...")
        camera.stop()            # VideoCaptureの解放とスレッド終了
        cv2.destroyAllWindows()  # 全 OpenCV ウィンドウの破棄
        pose_estimator.close()   # MediaPipe モデルのクローズ
        print("[*] システムは安全に停止されました。")


# ===========================================================================
# エントリーポイント
# ===========================================================================
if __name__ == "__main__":
    main()
