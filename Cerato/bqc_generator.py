from tkinter import filedialog
import ctypes
import os
import transitions
import shutil
import settings
import cv2
from tqdm import tqdm

def launch(filepath = None, temp_folder = 'temp', raw_folder = None):
    try: 
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    adj_path = filedialog.askopenfilename(
        title="Select input video",
        filetypes=[("Video File", "*.mp4")]    
    )
    if not os.path.exists(adj_path):
        print("FILE NOT FOUND")
        return

    save_path = filepath if filepath is not None else filedialog.asksaveasfilename(
        title="Select your where to save the output",
        filetypes=[("Video files", "*.mp4")] 
    )   

    style = None
    while style not in {1,2,3}:
        try:
            style = int(input("Enter style (between 1 to 3 - zoom in, zoom out, blend): "))
        except Exception as e:
            print(e)

    bqc_func_dir = {1: transitions.gbqc1, 2: transitions.gbqc2, 3: transitions.gbqc5}
    bqc_func = bqc_func_dir[style]
    time = float(input("Enter the time in seconds: "))
    

    os.makedirs(os.path.join(temp_folder, 'RAW'), exist_ok=True)
    os.makedirs(os.path.join(temp_folder, 'RB'), exist_ok=True)
    f1 = os.path.join(temp_folder, 'RAW', os.path.basename(adj_path))
    shutil.copy(adj_path, f1)
    print(raw_folder)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'mp4v' for .mp4
    out = cv2.VideoWriter(save_path, fourcc, settings.FPS, settings.VIDEO_SIZE)
    total_frames = int(settings.FPS * time)
    
    for i in tqdm(range(total_frames)):
        p = i/(total_frames-1)
        out.write(bqc_func(f1,f1,p,None,time, (0,0), raw_folder))
    
    out.release()
    
    