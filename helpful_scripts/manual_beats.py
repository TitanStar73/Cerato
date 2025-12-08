import numpy as np
import ast

video_path = "input.mp4"

scene_changes = ast.literal_eval(input("Enter in the values of each beat as a python list: "))

print(scene_changes)
print(len(scene_changes))
import pickle
name = input("Enter name: ")
with open(f"{name}.beats", "wb") as f:
    pickle.dump(scene_changes, f)
 
import os
os.system(f"ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 \"{name}.wav\"")

with open(f"{name}.transition", "w+") as f:
    f.write("""
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