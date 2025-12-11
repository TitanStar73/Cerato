print("This script will walk you through creating your own lut.")
print("This will replicate many filters and color corrections, however not all. Make sure your filter is LUT-compatible")
print("Some tips: ")
print("1) As much as possible avoid sending videos over a messaging platform. Instead use Google Drive, a flash drive or equivalent")
print("2) Export at the same qualitly level (HD) and use SDR.")

import numpy as np
from itertools import product
import cv2
from tqdm import tqdm
from scipy.spatial import KDTree
from itertools import product
import os

# --- Configuration ---
WIDTH = 1080
HEIGHT = 1920
FPS = 30
SECONDS_PER_IMAGE = 1 # Each unique image will be held for this duration
LUT_SIZE = 128        # The N value for our NxNxN LUT cube, less than the quantisation size (defined below, at 128)
OUTPUT_FILENAME = "synthetic_video.mp4"
FRAMES_PER_IMAGE = 30

PIXELS_PER_FRAME = WIDTH * HEIGHT
step = 255.0 / (LUT_SIZE - 1)
color_indices = product(range(LUT_SIZE), repeat=3)

master_colors = np.array(
    [
        (
            int(round(r_idx * step)), 
            int(round(g_idx * step)), 
            int(round(b_idx * step))
        )
        for r_idx, g_idx, b_idx in color_indices
    ], 
    dtype=np.uint8
)

synthetic_frames = []
np.random.seed(42)
# Loop 3 times to create 3 pairs of frames (6 total), adjust as needed for higher accuracy.
for _ in tqdm(range(3)):
    np.random.shuffle(master_colors)
    
    frame_a_pixels = master_colors[:PIXELS_PER_FRAME]
    np.random.shuffle(frame_a_pixels) 
    frame_a = frame_a_pixels.reshape((HEIGHT, WIDTH, 3))
    synthetic_frames.append(frame_a)
    
    frame_b_pixels = master_colors[-PIXELS_PER_FRAME:]
    np.random.shuffle(frame_b_pixels)
    frame_b = frame_b_pixels.reshape((HEIGHT, WIDTH, 3))
    synthetic_frames.append(frame_b)




fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, float(FPS), (WIDTH, HEIGHT))

if not video_writer.isOpened():
    print("Error: Could not open video writer.")
else:
    print(f"Writing video to {OUTPUT_FILENAME}...")
    for frame_rgb in synthetic_frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        for _ in range(FRAMES_PER_IMAGE):
            video_writer.write(frame_bgr)
            
    video_writer.release()
    print("Video writing complete.")


print(f"Synthetic video created. Run '{OUTPUT_FILENAME}' through your editor with the LUT settings")
print("Run it once with the LUT, save that file. Run it once with NO FILTERS and save that file too.")

def get_file_input(text):
    while True:
        k = input(text)
        if os.path.exists(k):
            return k
        print("File not found...")

unfiltered = get_file_input("Provide the filepath of the un-filtered video: ")
filtered = get_file_input("Provide the filepath of the filtered video: ")

cap2 = cv2.VideoCapture(unfiltered)
cap3 = cv2.VideoCapture(filtered)

for cap in [cap2, cap3]:
    if not cap.isOpened():
        raise FileNotFoundError

unfiltered_frames = []
filtered_frames = []

frame_nums = [14 + x*30 for x in range(6)] 
for frame_number in frame_nums:
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame2 = cap2.read()
    unfiltered_frames.append(frame2)
    
    cap3.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame3 = cap3.read()
    filtered_frames.append(frame3)    


cap2.release()
cap3.release()

import numpy as np

unfiltered_frames = [np.array(x, dtype=np.float64) for x in unfiltered_frames]
filtered_frames = [np.array(x, dtype=np.float64) for x in filtered_frames]

from collections import defaultdict
unfiltered_to_filtered_map = defaultdict(list) #In rgb

print("Building raw map...")
for i in range(6):
    pixels1 = unfiltered_frames[i].reshape(-1, 3)
    pixels2 = filtered_frames[i].reshape(-1, 3)

    for p1, p2 in zip(pixels1, pixels2):
        # Convert NumPy arrays to tuples to make them hashable (usable as dict keys), slow but DONT remove lol
        key = tuple(p1)[::-1] #Convert to rgb
        value = tuple(p2)[::-1] #Convert to rgb
        unfiltered_to_filtered_map[key].append(value)


print("Constructing true outpixels...")

for key in unfiltered_to_filtered_map:
    unfiltered_to_filtered_map[key] = [sum([k[i] for k in unfiltered_to_filtered_map[key]])/len(unfiltered_to_filtered_map[key]) for i in range(3)]
    
print("Constructing quantised mapping")
quantised_size = 128 #Change as needed, this is the max quantisation (lower = faster = lower quality, especially for gradients)


unfiltered_keys = list(unfiltered_to_filtered_map.keys())
kdtree = KDTree(unfiltered_keys)

print(f"Generating ideal {quantised_size}x{quantised_size}x{quantised_size} color grid...")
step = 255.0 / (quantised_size - 1)
quantized_grid = []
for r_idx, g_idx, b_idx in product(range(quantised_size), repeat=3):
    r = int(round(r_idx * step))
    g = int(round(g_idx * step))
    b = int(round(b_idx * step))
    quantized_grid.append((r, g, b))

distances, indices = kdtree.query(quantized_grid, k=1)

final_lut_map = {}
for i, quantized_pixel in enumerate(quantized_grid):
    closest_unfiltered_pixel = tuple(unfiltered_keys[indices[i]])
    filtered_pixel = unfiltered_to_filtered_map[closest_unfiltered_pixel]
    
    final_lut_map[quantized_pixel] = filtered_pixel


output_filename = f'{input("Enter in the name of your LUT: ")}.cube'

print(f"Writing data to '{output_filename}'...")
with open(output_filename, 'w') as f:
    # Write the header
    f.write(f'TITLE "Generated LUT {quantised_size}"\n')
    f.write(f'LUT_3D_SIZE {quantised_size}\n\n')
    
    for b_idx in range(quantised_size):
        for g_idx in range(quantised_size):
            for r_idx in range(quantised_size):
                r_quant = int(round(r_idx * step))
                g_quant = int(round(g_idx * step))
                b_quant = int(round(b_idx * step))
                
                r_out, g_out, b_out = final_lut_map[(r_quant, g_quant, b_quant)]
                
                #Normalize
                r_norm = r_out / 255.0
                g_norm = g_out / 255.0
                b_norm = b_out / 255.0
                
                f.write(f"{r_norm:.6f} {g_norm:.6f} {b_norm:.6f}\n")

print("Process complete.")

print("You may now delete all the files...")