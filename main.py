"""
===========================================================================
ハンドトラッキング + 仮想オブジェクト物理インタラクション
MediaPipe Hands で検出した指先で画面上のボールを押して動かすシステム
===========================================================================
前提ライブラリ:
    mediapipe == 0.10.21
    opencv-contrib-python == 4.11.0.86
    numpy == 1.26.4
    protobuf == 4.25.6
===========================================================================
操作方法:
    - 手をカメラに向けると骨格が表示される
    - 指先を仮想ボールに触れると押し出して動かせる
    - 'q' キーで終了 / 'r' キーでボールをリセット
===========================================================================
"""

import cv2
import sys
import time
import threading
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple


# ===========================================================================
# A. スレッドセーフな非同期カメラキャプチャクラス
# ===========================================================================
class ThreadedCamera:
    """
    専用デーモンスレッドでフレームを連続取得するクラス。
    推論負荷に関わらず常に最新フレームをバッファリングする。
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720) -> None:
        # DirectShow バックエンドで開く (MSMF/NVIDIA Broadcast との競合回避)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[!] DirectShow で開けませんでした。デフォルトバックエンドで再試行します...")
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[FATAL] カメラデバイス (src={src}) を開けません。"
                " USBカメラが接続されているか、デバイスインデックスを確認してください。"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def start(self) -> "ThreadedCamera":
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True, name="CameraThread")
        self.thread.start()
        return self

    def _update(self) -> None:
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self) -> Tuple[bool, np.ndarray | None]:
        with self.read_lock:
            if self.grabbed and self.frame is not None:
                return True, np.copy(self.frame)
            return False, None

    def stop(self) -> None:
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.cap.release()
        print("[*] カメラリソースを解放しました。")


# ===========================================================================
# B. 仮想ボール (物理演算付き)
# ===========================================================================
@dataclass
class VirtualBall:
    """
    2D 物理演算ボール。
    指先からの押し出し力を受けて加速し、減衰しながら画面内を動く。
    壁は完全反射する。
    """
    x: float
    y: float
    radius: int
    color: Tuple[int, int, int]
    vx: float = 0.0
    vy: float = 0.0
    # 物理パラメータ
    DAMPING:     float = 0.88   # 毎フレームの速度減衰率
    WALL_BOUNCE: float = 0.65   # 壁反射時の速度保持率
    MAX_SPEED:   float = 35.0   # 最大速度 (px/frame)

    def push(self, fx: float, fy: float, strength: float = 18.0) -> None:
        """
        指先 (fx, fy) から離れる方向に押し出し力を加える。
        """
        dx = self.x - fx
        dy = self.y - fy
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            dist = 1.0
        self.vx += (dx / dist) * strength
        self.vy += (dy / dist) * strength
        # 速度を上限にクランプ
        speed = np.hypot(self.vx, self.vy)
        if speed > self.MAX_SPEED:
            self.vx = self.vx / speed * self.MAX_SPEED
            self.vy = self.vy / speed * self.MAX_SPEED

    def update(self, w: int, h: int) -> None:
        """位置・速度を更新し、画面端で跳ね返る。"""
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.DAMPING
        self.vy *= self.DAMPING

        # 水平壁
        if self.x - self.radius < 0:
            self.x = float(self.radius)
            self.vx = abs(self.vx) * self.WALL_BOUNCE
        elif self.x + self.radius > w:
            self.x = float(w - self.radius)
            self.vx = -abs(self.vx) * self.WALL_BOUNCE

        # 垂直壁
        if self.y - self.radius < 0:
            self.y = float(self.radius)
            self.vy = abs(self.vy) * self.WALL_BOUNCE
        elif self.y + self.radius > h:
            self.y = float(h - self.radius)
            self.vy = -abs(self.vy) * self.WALL_BOUNCE

    def draw(self, frame: np.ndarray) -> None:
        """グロー効果付きでボールを描画する。"""
        cx, cy = int(self.x), int(self.y)
        speed = np.hypot(self.vx, self.vy)

        # ---- グロー (半透明の大きい円) ----
        glow_r = self.radius + int(speed * 0.8) + 10
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), glow_r, self.color, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # ---- 本体 ----
        cv2.circle(frame, (cx, cy), self.radius, self.color, -1)

        # ---- 境界線 (白) ----
        cv2.circle(frame, (cx, cy), self.radius, (255, 255, 255), 1)

        # ---- ハイライト ----
        h_off = self.radius // 3
        h_r   = max(self.radius // 5, 4)
        cv2.circle(frame, (cx - h_off, cy - h_off), h_r, (255, 255, 255), -1)


# ===========================================================================
# C. ユーティリティ
# ===========================================================================

# MediaPipe Hands の指先ランドマークID (親指~小指)
FINGERTIP_IDS: List[int] = [4, 8, 12, 16, 20]

# 指先の衝突判定半径 (px)
FINGERTIP_RADIUS: int = 16

# ボール定義 (color は BGR)
BALL_PRESETS = [
    {"color": (45,  105, 255), "radius": 45},   # 赤みオレンジ
]


def make_balls(w: int, h: int) -> List[VirtualBall]:
    """画面サイズに合わせてボールを均等配置する。"""
    # 単一ボールに変更
    positions = [
        (0.50, 0.50),
    ]
    balls = []
    for (rx, ry), preset in zip(positions, BALL_PRESETS):
        balls.append(VirtualBall(
            x=w * rx,
            y=h * ry,
            **preset,
        ))
    return balls


def draw_fingertip(frame: np.ndarray, fx: int, fy: int, touching: bool) -> None:
    """指先を視覚的に表示する。"""
    color_inner = (0, 255, 200) if touching else (255, 255, 255)
    color_ring  = (0, 180, 255) if touching else (180, 180, 180)
    cv2.circle(frame, (fx, fy), FINGERTIP_RADIUS, color_ring, 2)
    cv2.circle(frame, (fx, fy), 5, color_inner, -1)


# ===========================================================================
# D. メインループ
# ===========================================================================
def main() -> None:
    """
    ハンドトラッキング + 仮想ボール物理インタラクションのメインループ。
    """

    # -----------------------------------------------------------------------
    # 1. MediaPipe Hands の初期化
    # -----------------------------------------------------------------------
    print("[*] MediaPipe Hands を初期化中...")
    mp_hands         = mp.solutions.hands
    mp_drawing       = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands_estimator = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[*] MediaPipe Hands の初期化完了。")

    # -----------------------------------------------------------------------
    # 2. カメラ起動
    # -----------------------------------------------------------------------
    print("\n[*] カメラデバイスを起動中 (src=0, 1280x720)...")
    try:
        camera = ThreadedCamera(src=0, width=1280, height=720)
        camera.start()
    except RuntimeError as e:
        print(f"[!] {e}")
        hands_estimator.close()
        sys.exit(1)

    time.sleep(1.0)
    print("[*] 推論ループを開始します。")
    print("    ウィンドウ上で 'q' で終了、'r' でボールをリセットします。\n")

    # -----------------------------------------------------------------------
    # 3. 仮想ボールの初期化 (実際のフレームサイズで作成)
    # -----------------------------------------------------------------------
    _, init_frame = camera.read()
    h_cam = init_frame.shape[0] if init_frame is not None else 720
    w_cam = init_frame.shape[1] if init_frame is not None else 1280
    balls = make_balls(w_cam, h_cam)

    # -----------------------------------------------------------------------
    # 4. 推論・物理・描画ループ
    # -----------------------------------------------------------------------
    prev_time   = time.perf_counter()
    frame_count = 0

    try:
        while True:
            # --- フレーム取得 ---
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                print("[!] フレームのデコードに失敗しました。")
                break

            frame_count += 1

            # --- 左右反転 (鏡像にすることで自然な操作感) ---
            frame = cv2.flip(frame, 1)

            # --- BGR → RGB 変換 (MediaPipe 要件) ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            hands_results = hands_estimator.process(rgb_frame)
            rgb_frame.flags.writeable = True

            h, w = frame.shape[:2]

            # --- 指先座標の収集 ---
            fingertips: List[Tuple[int, int]] = []

            if hands_results.multi_hand_landmarks:
                for hand_lms in hands_results.multi_hand_landmarks:
                    # 手の骨格を描画
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_lms,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    # 全指先の画面座標を取得
                    for tip_id in FINGERTIP_IDS:
                        lm = hand_lms.landmark[tip_id]
                        fx = int(lm.x * w)
                        fy = int(lm.y * h)
                        fingertips.append((fx, fy))

            # --- 衝突判定 & 物理演算 ---
            # どの指先がどのボールに触れているかを記録
            touching_tips: set[int] = set()   # fingertips のインデックス

            for ball in balls:
                for i, (fx, fy) in enumerate(fingertips):
                    dx = ball.x - fx
                    dy = ball.y - fy
                    dist = np.hypot(dx, dy)
                    collision_dist = ball.radius + FINGERTIP_RADIUS
                    if dist < collision_dist:
                        # 衝突: 指先の方向にボールを押し出す
                        ball.push(fx, fy, strength=18.0)
                        touching_tips.add(i)

                ball.update(w, h)

            # --- ボールを描画 ---
            for ball in balls:
                ball.draw(frame)

            # --- 指先インジケーターを描画 (ボールの上に重ねる) ---
            for i, (fx, fy) in enumerate(fingertips):
                draw_fingertip(frame, fx, fy, touching=(i in touching_tips))

            # --- HUD ---
            curr_time = time.perf_counter()
            elapsed   = curr_time - prev_time
            fps       = 1.0 / elapsed if elapsed > 0 else 0.0
            prev_time = curr_time

            num_hands = len(hands_results.multi_hand_landmarks) if hands_results.multi_hand_landmarks else 0
            hud = f"FPS: {fps:.1f}  |  Hands: {num_hands}  |  'r': reset  'q': quit"
            cv2.putText(frame, hud, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   0,   0), 3)
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

            # --- 表示 ---
            cv2.imshow("VectorLine — Hand Physics Interaction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[*] 終了キー ('q') を検出しました。")
                break
            elif key == ord("r"):
                # ボールをリセット
                balls = make_balls(w, h)
                print("[*] ボールをリセットしました。")

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C による割り込みを検出しました。")

    finally:
        print("[*] リソースを解放中...")
        camera.stop()
        cv2.destroyAllWindows()
        hands_estimator.close()
        print("[*] システムは安全に停止されました。")


# ===========================================================================
# エントリーポイント
# ===========================================================================
if __name__ == "__main__":
    main()
