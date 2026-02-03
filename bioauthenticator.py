import cv2 
import mediapipe as mp
import os 
import numpy as np
import time # Added missing import

model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
baseOptions = mp.tasks.BaseOptions
handLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
visionrunningmode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=baseOptions(model_asset_path=model_path),
    running_mode=visionrunningmode.IMAGE,
    num_hands=1
)

detector = handLandmarker.create_from_options(options)
secret_sequence = ["PEACE", "FIST", "PEACE"]
current_step = 0
last_gesture = "NONE"
hold_start_time = 0
is_unlocked = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Critical: Initialize this every frame to avoid NameError
    detected_gesture = "NONE" 

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        lm = results.hand_landmarks[0]
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y
        
        if index_up and middle_up and not ring_up and not pinky_up:
            detected_gesture = "PEACE"
        elif not index_up and not middle_up and not ring_up and not pinky_up:
            detected_gesture = "FIST"

    # Sequence logic
    if not is_unlocked and current_step < len(secret_sequence):
        target = secret_sequence[current_step]
        if detected_gesture == target:
            if last_gesture != target:
                hold_start_time = time.time()
                last_gesture = target
            
            if time.time() - hold_start_time > 1.0:
                current_step += 1
                last_gesture = "NONE"
                if current_step == len(secret_sequence):
                    is_unlocked = True
        else:
            last_gesture = "NONE"

    # Fixed UI Logic
    color = (0, 255, 0) if is_unlocked else (0, 0, 255)
    status_text = "YOU SHALL PASS" if is_unlocked else "YOU SHALL NOT PASS"
    
    cv2.rectangle(frame, (0, 0), (w, h), color, 10)
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if not is_unlocked:
        # Show progress
        cv2.putText(frame, f"STEP: {current_step}/{len(secret_sequence)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"HOLD: {secret_sequence[current_step]}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Bioauthentication", frame)
    if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
        break

detector.close()
cap.release()
cv2.destroyAllWindows()