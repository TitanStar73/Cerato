import cv2
import numpy as np

video_path = "input.mp4"
threshold = 0.1  # tweak this; higher = less sensitive

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
prev_frame = None
frame_number = 0
scene_changes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        non_zero_count = np.sum(diff > 30)  # count of significantly changed pixels
        if non_zero_count > threshold * gray.size / 100:  # percent change
            timestamp = frame_number / fps
            scene_changes.append(timestamp)
    
    prev_frame = gray
    frame_number += 1

cap.release()
print(scene_changes)
print(len(scene_changes))
import pickle
name = input("Enter name: ")
with open(f"{name}.beats", "wb") as f:
    pickle.dump(scene_changes, f)
 
import os
os.system(f"ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 \"{name}.wav\"")

with open(f"{name}.transition", "w+") as f:
    f.write("""//Additional Info
{
    "linear_cut": {
        "lowest_bpm": 1,
        "highest_bpm": 20000,
        "weight": 10,
        "beats": 1,
        "intro_priority": False,
        "boosts": {},
        "boosts_long_term": {}
    }
}""")