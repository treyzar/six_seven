import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from collections import deque
import time
import urllib.request

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Создаем лэндмаркер с коллбэком
class HandTracker:
    def __init__(self):
        self.particles = []
        self.motion_trails = {"Left": deque(maxlen=24), "Right": deque(maxlen=24)}
        self.flash_alpha = 0.0
        self.combo = 0
        self.best_combo = 0
        self.combo_step_sec = 0.9
        self.combo_grace_sec = 0.35
        self.combo_hold_start = None
        self.last_pair_seen = 0.0
        self.combo_progress = 0.0
        self.streak_text = ""
        self.streak_text_until = 0.0
        self.last_results = {"landmarks": None, "handedness": None}
        
        def result_callback(result, output_image, timestamp_ms):
            self.last_results["landmarks"] = result.hand_landmarks if result.hand_landmarks else []
            self.last_results["handedness"] = result.handedness if result.handedness else []
        
        model_path = self._resolve_model_path()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=result_callback
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _resolve_model_path(self):
        # Приоритет: переменная окружения, затем файл рядом со скриптом, затем системные временные папки.
        env_path = os.environ.get("HAND_LANDMARKER_MODEL")
        local_model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
        candidates = [
            env_path,
            str(local_model_path),
            "/tmp/hand_landmarker.task",
            r"C:\tmp\hand_landmarker.task",
        ]

        for path in candidates:
            if path and Path(path).is_file():
                return path

        # Для удобства первого запуска пробуем автоматически скачать модель.
        try:
            print("Файл модели не найден. Скачиваю hand_landmarker.task...")
            urllib.request.urlretrieve(MODEL_URL, str(local_model_path))
            if local_model_path.is_file():
                print(f"Модель сохранена: {local_model_path}")
                return str(local_model_path)
        except Exception:
            pass

        raise FileNotFoundError(
            "Не найден файл модели hand_landmarker.task.\n"
            "Автозагрузка не удалась. Положите файл в папку проекта (рядом с hand_tracker.py) "
            "или задайте переменную HAND_LANDMARKER_MODEL с полным путем."
        )

    def _draw_motion_trails(self, overlay):
        for hand_label, points in self.motion_trails.items():
            if len(points) < 2:
                continue

            base_color = (90, 255, 180) if hand_label == "Left" else (255, 180, 90)
            trail = list(points)

            for i in range(1, len(trail)):
                glow = i / len(trail)
                color = tuple(int(channel * glow) for channel in base_color)
                thickness = max(1, i // 4 + 1)
                cv2.line(overlay, trail[i - 1], trail[i], color, thickness)

    def _register_combo_hit(self, now, left_center, right_center):
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        self.flash_alpha = 0.24
        self.streak_text_until = now + 1.2

        if self.combo < 3:
            self.streak_text = "GOOD!"
        elif self.combo < 6:
            self.streak_text = "NICE STREAK!"
        elif self.combo < 10:
            self.streak_text = "SUPER 67!"
        else:
            self.streak_text = "LEGEND 67!"

        center_x = int((left_center[0] + right_center[0]) / 2)
        center_y = int((left_center[1] + right_center[1]) / 2)
        burst_size = min(8 + self.combo * 2, 28)
        burst_color = (100, 255, 255) if self.combo % 2 == 0 else (255, 120, 255)
        for _ in range(burst_size):
            self.particles.append({
                'x': center_x + np.random.randint(-24, 24),
                'y': center_y + np.random.randint(-24, 24),
                'vx': np.random.uniform(-3.5, 3.5),
                'vy': np.random.uniform(-4.2, -0.8),
                'life': 34,
                'color': burst_color
            })

    def _draw_combo_hud(self, frame):
        cv2.putText(
            frame,
            f"COMBO: x{self.combo}",
            (20, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (90, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            f"BEST: x{self.best_combo}",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 180, 120),
            2,
        )

        bar_x, bar_y = 20, 94
        bar_w, bar_h = 340, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 2)
        fill_w = int(bar_w * np.clip(self.combo_progress, 0.0, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + fill_w - 2, bar_y + bar_h - 2), (80, 240, 255), -1)

        cv2.putText(
            frame,
            "Hold both hands (6 + 7) to build combo",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            2,
        )

    def _apply_super_overlay(self, overlay, now):
        pulse = 0.5 + 0.5 * np.sin(now * 10.0)
        tint = np.full_like(overlay, (50, int(70 + 90 * pulse), int(120 + 90 * (1 - pulse))))
        cv2.addWeighted(tint, 0.12, overlay, 0.88, 0, overlay)
        if self.combo >= 6:
            self.draw_glowing_text(overlay, "SUPER 67", (overlay.shape[1] - 530, 88), (100, 255, 255))

    def create_particles(self, x, y, color):
        for _ in range(5):
            self.particles.append({
                'x': x + np.random.randint(-20, 20),
                'y': y + np.random.randint(-20, 20),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-3, -1),
                'life': 30,
                'color': color
            })
    
    def draw_glowing_text(self, img, text, pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(5, 0, -1):
            alpha = 0.3 - i * 0.05
            thickness = 8 + i * 2
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.putText(img, text, pos, font, 3, glow_color, thickness)
        cv2.putText(img, text, pos, font, 3, color, 8)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Конвертируем в формат MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            # Обрабатываем кадр
            self.landmarker.detect_async(mp_image, timestamp_ms)
            
            overlay = frame.copy()

            hand_landmarks_list = self.last_results["landmarks"]
            handedness_list = self.last_results["handedness"]
            now = time.time()
            detected_hands = set()
            palm_centers = {}

            if hand_landmarks_list:
                for idx, hand_landmarks in enumerate(hand_landmarks_list):
                    h, w, _ = frame.shape

                    handedness = handedness_list[idx] if idx < len(handedness_list) else None
                    hand_label = handedness[0].category_name if handedness else "Right"
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                    is_left = hand_label == "Left"
                    text = "7" if is_left else "6"
                    color = (255, 100, 255) if is_left else (100, 255, 255)

                    for i in range(0, 21, 2):
                        cv2.circle(overlay, landmarks[i], 8, color, -1)

                    # HAND_CONNECTIONS: (0,1),(1,2),(2,3),(3,4), и т.д.
                    connections = [
                        (0,1),(1,2),(2,3),(3,4),
                        (0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),
                        (0,13),(13,14),(14,15),(15,16),
                        (0,17),(17,18),(18,19),(19,20),
                        (5,9),(9,13),(13,17)
                    ]
                    
                    for connection in connections:
                        start = landmarks[connection[0]]
                        end = landmarks[connection[1]]
                        cv2.line(overlay, start, end, color, 3)

                    palm_x = int(np.mean([landmarks[i][0] for i in [0, 5, 9, 13, 17]]))
                    palm_y = int(np.mean([landmarks[i][1] for i in [0, 5, 9, 13, 17]]))
                    self.motion_trails[hand_label].append((palm_x, palm_y))
                    palm_centers[hand_label] = (palm_x, palm_y)
                    detected_hands.add(hand_label)

                    self.draw_glowing_text(overlay, text, (palm_x - 40, palm_y + 20), color)

                    if np.random.random() > 0.7:
                        self.create_particles(palm_x, palm_y, color)

            if "Left" not in detected_hands:
                self.motion_trails["Left"].clear()
            if "Right" not in detected_hands:
                self.motion_trails["Right"].clear()

            pair_active = "Left" in detected_hands and "Right" in detected_hands
            if pair_active:
                self.last_pair_seen = now
                if self.combo_hold_start is None:
                    self.combo_hold_start = now
                    self.combo_progress = 0.0
                elapsed = now - self.combo_hold_start
                while elapsed >= self.combo_step_sec:
                    self.combo_hold_start += self.combo_step_sec
                    elapsed = now - self.combo_hold_start
                    self._register_combo_hit(now, palm_centers["Left"], palm_centers["Right"])
                self.combo_progress = min(1.0, elapsed / self.combo_step_sec)
            else:
                if now - self.last_pair_seen > self.combo_grace_sec:
                    self.combo = 0
                    self.combo_hold_start = None
                    self.combo_progress = 0.0

            combo_visual_mode = self.combo >= 2
            if combo_visual_mode:
                self._draw_motion_trails(overlay)
                self._apply_super_overlay(overlay, now)

            for particle in self.particles[:]:
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['life'] -= 1
                
                if particle['life'] > 0:
                    size = max(1, particle['life'] // 6)
                    cv2.circle(overlay, (int(particle['x']), int(particle['y'])), size, particle['color'], -1)
                else:
                    self.particles.remove(particle)

            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            if self.flash_alpha > 0:
                flash_layer = np.full_like(frame, 255)
                cv2.addWeighted(flash_layer, self.flash_alpha, frame, 1 - self.flash_alpha, 0, frame)
                self.flash_alpha = max(0.0, self.flash_alpha - 0.04)

            self._draw_combo_hud(frame)

            if now < self.streak_text_until:
                self.draw_glowing_text(frame, self.streak_text, (frame.shape[1] // 2 - 240, 150), (120, 255, 255))

            cv2.putText(
                frame,
                "q: exit | r: reset combo",
                (20, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
            )

            cv2.imshow('Hand Tracker - 6 & 7', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.combo = 0
                self.combo_hold_start = None
                self.combo_progress = 0.0
                self.streak_text = "RESET"
                self.streak_text_until = now + 0.8
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
