import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
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
        
        import time
        
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
            
            if hand_landmarks_list:
                for idx, (hand_landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
                    h, w, _ = frame.shape
                    
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                    
                    is_left = handedness[0].category_name == "Left"
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
                    
                    self.draw_glowing_text(overlay, text, (palm_x - 40, palm_y + 20), color)
                    
                    if np.random.random() > 0.7:
                        self.create_particles(palm_x, palm_y, color)
            
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
            
            cv2.imshow('Hand Tracker - 6 & 7', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
