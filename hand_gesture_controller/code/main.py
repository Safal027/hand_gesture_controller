import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pyautogui
import math
import numpy as np
import time
from pathlib import Path

screen_w, screen_h = pyautogui.size()

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

latest_result = None


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

path_to_model = f"{Path(__file__).parent.parent}" + \
    r"\hand_landmarker_model\hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=path_to_model),
    num_hands=1,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

detector = vision.HandLandmarker.create_from_options(options)

start_time = time.time()

prev_time = 0
count = 0

frame_reduction = 90

while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

    timestamp_ms = int((time.time() - start_time) * 1000)
    results = detector.detect_async(mp_image, timestamp_ms)

    h, w, _ = img.shape

    cv2.rectangle(img, (frame_reduction, frame_reduction),
                  (w-frame_reduction, h-frame_reduction), (255, 255, 255), 2)

    if latest_result and latest_result.hand_landmarks:

        for hand_landmarks in latest_result.hand_landmarks:
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    curr_time = time.time()
    fps = int(1/(curr_time-prev_time))
    prev_time = curr_time

    if count % 10 == 0:
        fps_slowed = fps
    cv2.putText(img, str(fps_slowed), (10, 30), 1, 3, (255, 0, 255), 3)
    count = count+1

    cv2.imshow("", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
