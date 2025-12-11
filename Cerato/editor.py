from settings import *
"""
Potential Improvements Post 1.0.0:
 - More accurate scene splitter (currently uses keyframes)
 - MGE:
    - Contour wisps (better implementation, more tunable)
    - slight wobble back and forth
    - Faster hot-reload (lowering fps/resolution)
    - Complete overhaul (easier video selection, stackable filters)
    - Merge with trim-cropper
 - Extend support to other format (.mov, .avi, .jpg, .jpeg, etc.) Maybe just convert all at the start of the script
 - Anti-compression during generation of syntethic images (or more syntethic images) to ensure proper colour space coverage of the generated LUTS (currently do bad with gradients)
 - Implement the LLM-based font/color/glow-color picker. Current method is deterministic, and the caption_json_ai() func attempts to implement this but results are poor
 - Better beat_times
 - Add a fuzzy logic based project selector, all projects have proj.cerato file in their directory
 - Sound effects on transitions?

Cerato (For Ceratopipra, i.e. manakin)
"""

import name #Creates the cool rainbow logo
print(r"""
A video editing tool
╔╗ ╦ ╦  ┌─┐┬─┐┬ ┬┌─┐┬ ┬
╠╩╗╚╦╝  ├─┤├┬┘│ │└─┐├─┤
╚═╝ ╩   ┴ ┴┴└─└─┘└─┘┴ ┴
      
$ Lets go step by step, steps already done will be skipped. You should delete the folder related to a step, if you want to do it again.
$ Songs should be imported to songs/ relative to this script
$ Automatically saves work as done (no repeating yourself!)

$ Version 1.2.1-release
""".replace("$", "\033[31m#\033[0m")) #Makes '$'' into a red '#'

input("Click enter to continue...")

import os
if os.path.exists(f"name.cookie"): 
    with open("name.cookie") as f:
        DEFAULT_PROJECT = f.read()
    print(f"Defaults to {DEFAULT_PROJECT} | ", end = "")

PROJECT_NAME = "" if DEBUG_MODE else input("Enter project name: ").lower().replace(" ", "_")

if PROJECT_NAME == "":
    if os.path.exists(f"name.cookie"):
        PROJECT_NAME = DEFAULT_PROJECT
        print(f"Prior project {PROJECT_NAME} selected")
    else:
        print("Invalid name!")
        exit()
else:
    with open("name.cookie", "w+") as f:
        f.write(PROJECT_NAME)

print("Importing libraries...")
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from rapidfuzz import process
from random import shuffle
import glob
import shutil
import librosa
import numpy as np
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips, VideoFileClip
from moviepy.audio.fx import AudioFadeOut
from PIL import Image, ImageDraw
import pickle
import settings
import rembg
session = rembg.new_session(settings.RB_MODEL)
def remove(stuff):
    global session
    return rembg.remove(stuff, session)


from tqdm import tqdm
import cv2
import transition_registry
import random
import re
import json
import json_writer
json_columns = ["Start of block", "end of in-transition", "start of out transition", "end of block", "inward transition function", "outward transition function", "bpm", "filepath", "beats"]
import ffxcc
from pydub import AudioSegment
import tkinter as tk
from tkinter import filedialog
import ctypes
import caption
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import time
import keyboard
import subprocess
import scene_pack_splitter
import paper_animator
import transition_regs_adv

os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}", exist_ok = True)
with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/proj.cerato", "w+") as f:
    f.write("")

JUST_HOOK = lambda x:x
JUST_EDIT = lambda x:x
FILTER_SETTINGS = {"vhs_on_edit": True, "vhs_on_hook": True, "overlays_on_edit": True, "overlays_on_hook":True, "captions_on_hook":True, "paper_animation_request":False, "bqc_request":False}

#Adds watermark settings
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/watermark.png"):
    import watermark
    watermark.set_up_filter(f"{PROJECT_FOLDER}/{PROJECT_NAME}/watermark.png",f"{PROJECT_FOLDER}/{PROJECT_NAME}/mask1.png", f"{PROJECT_FOLDER}/{PROJECT_NAME}/mask2.png")
    watermark_filter = watermark.main_filter
    print("watermark found!")
else:
    print("Warning: No watermark (and respective mask1 and mask2 for glows)")
    watermark_filter = lambda x:x


#Sets filter settings
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/filters.json"):
    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/filters.json", "r") as f:
        FILTER_SETTINGS = json.load(f)
    
    del JUST_HOOK
    del JUST_EDIT

    def JUST_HOOK(x):
        y = x.copy()
        if FILTER_SETTINGS["vhs_on_hook"]:
            y = ffxcc.light_vhs_filter(y, FPS=FPS)
        if FILTER_SETTINGS["overlays_on_hook"]:
            y = ffxcc.apply_borderless_effect(watermark_filter(y))

        return y

    def JUST_EDIT(x):
        y = x.copy()
        if FILTER_SETTINGS["vhs_on_edit"]:
            y = ffxcc.light_vhs_filter(y, FPS=FPS)
        if FILTER_SETTINGS["overlays_on_edit"]:
            y = ffxcc.apply_borderless_effect(watermark_filter(y))

        return y

    if FILTER_SETTINGS["paper_animation_request"]:
        print("\nLaunching paper animator...")
        paper_animator.launch(f"{PROJECT_FOLDER}/{PROJECT_NAME}/paper_animation.mp4", f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/PAPER_STILLS")
        FILTER_SETTINGS["paper_animation_request"] = False
        with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/filters.json", "w") as f:
            json.dump(FILTER_SETTINGS, f, indent=4)
  
        print("Paper animation done.")
        print(f"Stills saved in {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/PAPER_STILLS/. Can be used for priority background changes...")
        print(f"Paper animation saved as: {PROJECT_FOLDER}/{PROJECT_NAME}/paper_animation.mp4, can be used for outros")
    
    if FILTER_SETTINGS["bqc_request"]:
        print("\nLaunching background quick change animator...")
        os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/BQC", exist_ok=True)
        os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/BQC/RENDER", exist_ok=True)
        
        import bqc_generator
        bqc_generator.launch(f"{PROJECT_FOLDER}/{PROJECT_NAME}/bqc_animation.mp4",f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/BQC/RENDER", raw_folder = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW")
        FILTER_SETTINGS["bqc_request"] = False
        with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/filters.json", "w") as f:
            json.dump(FILTER_SETTINGS, f, indent=4)
  
        print("bqckground quick change animation done.")
        print(f"saved as: {PROJECT_FOLDER}/{PROJECT_NAME}/bqc_animation.mp4, can be used for RAW media")
    


#Creating SONG/ and subfiles
os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG", exist_ok = True)

if ALWAYS_ASK_FOR_SONG or not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav"):
    music_library = [f"{os.path.basename(f)}" for f in glob.glob("songs/*.wav")]
    music_library = [f"<ansired>{x}</ansired>" if os.path.exists("songs/" + x.replace(".wav", ".beats")) else x for x in music_library]

    music_library2 = list(music_library)
    music_library2.append("Random")

    class FuzzyMusicCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text
            if text.strip():
                matches = process.extract(
                    text,
                    music_library2,
                    limit=10,  # number of suggestions
                    score_cutoff=40  # minimum similarity score
                )
                for match, score, _ in matches:
                    yield Completion(match.replace('<ansired>','').replace('</ansired>',''), start_position=-len(text), display=HTML(match))

    completer = FuzzyMusicCompleter()
    print("\033[31mRed\033[0m means a template/transition-registry is available for the song")
    print("'Random' will take a random song from the catalouge")
    song = prompt("Search music: ", completer=completer)

    if song.lower() in {'random', 'r'}:
        random.seed()
        shuffle(music_library)
        song = music_library[0]

    song_file = f"songs/{song}"

    ###song.txt -> either does not exist, 
    # exists as a single float -> the point between hook and main
    # exists in the form 'number1, number2' -> number1 is the point between hook and main and number2 is the point between main and outro  

    if not os.path.exists(f"songs/{song}.txt"):
        shutil.copy(song_file, f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav")
    else:
        with open(f"songs/{song}.txt") as f:
            song_data = f.read()

        try:
            x = int(1000 * float(song_data))
            inp_song = AudioSegment.from_wav(song_file)
        
            hook_song = inp_song[:x]
            main_song = inp_song[x:]

            hook_song.export(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook.wav", format="wav")
            main_song.export(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav", format="wav")
        except ValueError: #i.e. song data is NOT a float
            x1,x2 = [int(1000 * float(x)) for x in song_data.split(",")]
            inp_song = AudioSegment.from_wav(song_file)
        
            hook_song = inp_song[:x1]
            main_song = inp_song[x1:x2]
            outro_song = inp_song[x2:]

            hook_song.export(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook.wav", format="wav")
            main_song.export(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav", format="wav")
            outro_song.export(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro.wav", format="wav")
            

    print(f"Song {song_file} added")


#Creates beats
if ALWAYS_ASK_FOR_SONG or not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/beats.pkl"):
    print("Calculating beats...")
    try:
        shutil.copy(song_file.replace(".wav", ".beats"), f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/beats.pkl")
        print("Imported Template Beats")
    except:
        pass
    
    try:
        shutil.copy(song_file.replace(".wav", ".transition"), f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/transition_edit.json")
        print("Imported Template Transitions")
    except:
        pass


    try:
        shutil.copy(song_file.replace(".wav", ".template"), f"{PROJECT_FOLDER}/{PROJECT_NAME}/template.pkl")
        print("Imported Template")
    except:
        pass


if ALWAYS_ASK_FOR_SONG or not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/beats.pkl"): #Create beats if not imported from a template
    y, sr = librosa.load(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav", sr=None, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) 


    tempo = float(tempo)

    print(f"Estimated tempo: {tempo:.2f} BPM")
    print("Beat timestamps (in seconds):")
    for t in beat_times:
        print(f"{t:.2f}")


    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/beats.pkl", "wb") as f:
        pickle.dump(beat_times, f)
else:
    print("Loaded beats...")
    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/beats.pkl", "rb") as f:
        beat_times = pickle.load(f)


#Adds a visualiser for the beats
if not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/visualised.mp4"):
    print("Creating visualisation, within SONG folder...")

    AUDIO_FILE = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav"
    OUTPUT_VIDEO = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/visualised.mp4"
    WIDTH, HEIGHT = 720, 720
    FPS = 30
    BG_COLOR = (10, 10, 20)   # background color (dark)
    CIRCLE_COLOR = (255, 100, 80)  # circle color
    BASE_RADIUS = 80          # radius (pixels) at rest
    PEAK_RADIUS = 300         # radius (pixels) right at beat peak
    DECAY_SECONDS = 0.25      # how fast the pulse shrinks after a beat
    

    beat_times = np.asarray(sorted(beat_times))


    audio_clip = AudioFileClip(AUDIO_FILE)
    duration = audio_clip.duration


    def last_beat_before(t):
        idx = np.searchsorted(beat_times, t, side='right') - 1
        if idx >= 0:
            return beat_times[idx]
        return None

    def make_frame(t):
        # background image via PIL
        img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # find last beat and compute radius factor
        lb = last_beat_before(t)
        if lb is None:
            radius = BASE_RADIUS
        else:
            dt = t - lb
            # If dt is very small (on beat), radius ~ PEAK_RADIUS.
            # After the beat, radius decays exponentially toward BASE_RADIUS.

            if dt < 0:
                dt = 0

            # exponential decay from 1 -> 0 with DECAY_SECONDS time constant
            decay = np.exp(-dt / DECAY_SECONDS)
            radius = BASE_RADIUS + (PEAK_RADIUS - BASE_RADIUS) * decay

        # draw pulsing circle in center
        cx, cy = WIDTH // 2, HEIGHT // 2
        r = int(radius)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=CIRCLE_COLOR)

        frame = np.array(img)
        return frame


    video_clip = VideoClip(make_frame, duration=duration)
    video_clip = video_clip.with_fps(FPS)
    # attach audio
    video_clip = video_clip.with_audio(audio_clip)


    print(f"{len(beat_times)} beats found.")
    print("Rendering video — this may take a while depending on length and machine speed...")
    video_clip.write_videofile(
        OUTPUT_VIDEO,
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        preset="medium",
        threads=4,
        ffmpeg_params=["-crf", "18"]
    )
else:
    print("Visualisation alread made!")


TRANSITIONS = dict(transition_registry.TRANSITIONS)
for key in TRANSITIONS:
    TRANSITIONS[key]["highest_bpm"] *= 2
TRANSITIONS["default"]["highest_bpm"] = float("inf")
TRANSITIONS["default"]["lowest_bpm"] = 0

if not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/transition_packs.json"):
    transitions_packs = ["DEFAULT"]

    result = transition_regs_adv.get_transitions()
    keys = list(result.keys()) + ['DEFAULT']

    music_library2 = ["exit", "all"] + keys + ["REM " + key for key in keys]

    class FuzzyMusicCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text
            if text.strip():
                matches = process.extract(
                    text,
                    music_library2,
                    limit=10,  # number of suggestions
                    score_cutoff=40  # minimum similarity score
                )
                for match, score, _ in matches:
                    yield Completion(match.replace('<ansired>','').replace('</ansired>',''), start_position=-len(text), display=HTML(match))

    stack = ["DEFAULT"]

    completer = FuzzyMusicCompleter()
    print("We will now construct the transition registry...")
    print("You can add a pack by typing in the pack name. Remove it by typing 'REM pack_name'")
    print("Enter 'exit' to save and exit.\n\n")
    print("Transition registry is applied top to bottom, with bottom overwriting top")
    print("'all' adds all packs to the stack")

    def simplify_stack(stack):
        new_stack = []
        for item in stack[::-1]:
            if item not in new_stack:
                new_stack.append(item)

        return new_stack[::-1]
    
        
    while True:
        print("\nCurrent stack:")
        for item in stack:
            print(f"  ~ {item}")
        print("")
        song = prompt(">>> ", completer=completer).upper()

        if song == "EXIT":
            break

        if song == "ALL":
            for key in keys:
                stack.append(key)
        elif song in keys:
            stack.append(song)
        elif song.startswith("REM ") and song[4:] in keys:
            stack = [item for item in stack if item != song[4:]]
        else:
            print("Not a valid pack")

        stack = simplify_stack(stack)

        
        
    transitions_packs = list(stack)

    if input("Would you like to nullify any temporal boosts?: ").lower() in {'yes', 'y'}:
        transitions_packs.append("NO_BOOSTS")

    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/transition_packs.json", "w") as f:
        json.dump(transitions_packs, f, indent=4)

if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/transition_packs.json"):

    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/transition_packs.json") as f:    
        obj = json.load(f)

    if "DEFAULT" not in obj:
        TRANSITIONS = dict()
        
    result = transition_regs_adv.get_transitions()
    for key in obj:
        if key.upper() in result:
            for key2 in result[key.upper()]:
                TRANSITIONS[key2] = result[key.upper()][key2]

    if "NO_BOOSTS" in obj:
        for key in TRANSITIONS:
            TRANSITIONS[key]["boosts_long_term"] = {key3:TRANSITIONS[key]["boosts_long_term"][key3] for key3 in TRANSITIONS[key]["boosts_long_term"] if TRANSITIONS[key]["boosts_long_term"][key3] < 0}

            TRANSITIONS[key]["boosts"] = {key3:TRANSITIONS[key]["boosts"][key3] for key3 in TRANSITIONS[key]["boosts"] if TRANSITIONS[key]["boosts"][key3] < 0}

else:
    print("\nNo transitions_packs.json, so defaulting to DEFAULT\n")            


os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA", exist_ok = True)
os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", exist_ok = True)
os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB_HIGHLIGHT", exist_ok = True)

number_of_media = len([f for f in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW") if os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", f))])
os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW/HIST", exist_ok = True)
with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW/HIST/readme.txt", "w+") as file:
    file.write("This /HIST/ folder is for storing the prior versions of edits to the /RAW/ folder. Files here may be deleted if you want. However it is highly encouraged to atleast keep the first history version so that further edits use the original image...")


print("")
if number_of_media != 0:
    print("Great you've added Media!")

if input("Would you like to load some media directly? (You can trim and crop the video) Enter 'y': ").lower() in {'y', 'yes'}:
    scene_pack_splitter.launch_adv(
        file_func=lambda x:f"{x}_{random.randint(int(1e9),int(1e10-1))}_trim.mp4",
        default_dir=f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW",
        default_pic=lambda x:f"{x}_{random.randint(int(1e9),int(1e10-1))}_screenshot.png"
    )
    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt", "w+") as file:
        file.write("1")

if input("Would you like to convert photos to a videos?? (So you can apply temporal edits?): ").lower() in {'y','yes'}:
    from tkinter import filedialog
    scene_pack_splitter.make_dpi_aware()
    print("Great you will be asked for a filepath, then the output length. Finally you can decide where to save it. If anything is invalid, it will exit...")
    input("Click enter to continue...")
    while True:
        img_path = filedialog.askopenfilename(title="Pick your image (click cancel to exit)",filetypes=[("Image File", "*.png *.jpg *.jpeg"), ("All Files", "*.*")])
        if not os.path.exists(img_path):
            break
        print("Image path: " + img_path)
        try:
            vid_length = float(input("Enter the video length: "))
        except:
            break
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", 
                                                 initialdir = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", 
                                                 initialfile = f"{os.path.splitext(os.path.basename(img_path))[0]}_vid.mp4", 
                                                 filetypes=[("Video File", f"*.mp4"), ("All Files", "*.*")])

        print("Save path: " + save_path)

        os.system(f'ffmpeg -y -hide_banner -loglevel error -loop 1 -i "{img_path}" -c:v libx264 -t {vid_length} -pix_fmt yuv420p "{save_path}"')


    print("Exited IMG -> VID... continuing with the rest of the script...")


if number_of_media < len(beat_times)//10:
    print(f"Blud there are {len(beat_times)} beats, you've only added {number_of_media} items. Atleast {len(beat_times)//10}, but target atleast {len(beat_times)//3} and upto {len(beat_times)}. Yes videos count for more, since they will be played duration-based and may take up more than a beat. Add more too {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW")
    exit()


## Manage edits
#functions using this must be run before the 'Managing RBs' section of the script
def update_file(og_path, force_i = None):
    print(f"Updating {og_path} w/ {force_i}")
    """Takes in filename -> moves it to HIST location & updates redo.txt"""

    directory = os.path.dirname(og_path) 
    filename, extension = os.path.splitext(os.path.basename(og_path))

    new_directory = os.path.join(directory, "HIST")

    if type(force_i) == int:
        new_filename = os.path.join(new_directory, f"{filename}_{force_i}{extension}")
        if os.path.exists(new_filename):
            raise Exception("Hist file already exists!!")
    else:
        for i in range(1,10000):
            new_filename = os.path.join(new_directory, f"{filename}_{i}{extension}")
            if not os.path.exists(new_filename):
                break

    shutil.copy2(og_path, new_filename)

    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt", "w+") as file:
        file.write("1")

    return str(new_filename)

def get_og_file_path(my_path, needs_exist = True, force_i = None):
    """Takes in the current png and returns the original HIST file"""

    directory = os.path.dirname(my_path) 
    filename, extension = os.path.splitext(os.path.basename(my_path))

    new_directory = os.path.join(directory, "HIST")

    if type(force_i) == int:
        return os.path.join(new_directory, f"{filename}_{force_i}{extension}")

    new_filename = os.path.join(new_directory, f"{filename}_1{extension}")

    if os.path.exists(new_filename) or not needs_exist:
        return new_filename
    return my_path

def og_file_path_exists(my_path, force_i = 1):
    """Takes in the current png and returns if the original HIST file exists"""

    directory = os.path.dirname(my_path) 
    filename, extension = os.path.splitext(os.path.basename(my_path))

    new_directory = os.path.join(directory, "HIST")

    new_filename = os.path.join(new_directory, f"{filename}_{force_i}{extension}")

    return os.path.exists(new_filename)


## Increase FPS, CC and more
if not DISABLE_EDITS and input("Edit your media?? (y for 'yes', default is no): ").lower().strip() in {'yes', 'y'}:
    #Boost FPS
    for filename in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW"):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", filename)
            
            if og_file_path_exists(video_path, force_i=1):
                print(f"fps done already for {video_path}")
                continue #Means fps already done
            if og_file_path_exists(video_path, force_i=0):
                raise Exception("Critical error!! Appears you stopped the script midway? Try deleting contents of MEDIA/RAW/HIST/")
            
            update_file(video_path, force_i = 0)           
            #get og frame count
            cap = cv2.VideoCapture(get_og_file_path(video_path, force_i=0))
            if not cap.isOpened():
                raise Exception(f"Error: Could not open video file {video_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            initial_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            #get new frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Error: Could not open video file {video_path}")
            frame_count2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count >= FPS * MAX_MEDIA_USAGE or frame_count2 >= FPS * MAX_MEDIA_USAGE: #i.e. at most of 1.5 s per MEDIA dispalyed
                continue #Never needed a FPS boost
            if frame_count2/frame_count >= 3:
                continue #Already got a fps boost
            
            target_fps = int(min(((FPS * MAX_MEDIA_USAGE)/frame_count),4) * initial_fps * 1.2) + 1 #Max 2.5x fps
            
            #Using standard fps because vid_load in transitions can't support optimized fps counts. Might be better this ways anyways so external media players dont break
            if target_fps <= 30:
                target_fps = 30
            elif target_fps <= 60:
                target_fps = 60
            else:
                target_fps = 120

            command = f'ffmpeg -i "{get_og_file_path(video_path, force_i = 0)}" -y -vf "minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -c:v libx264 -preset medium -crf 18 -c:a copy "{video_path}"'
            print(command)
            os.system(command)
            update_file(video_path, force_i=1)


    for filename in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW"):
        if '.' not in filename:
            continue
        print(f"file: '{filename}'")
        mypath = os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", filename)

        if not og_file_path_exists(mypath, force_i=1):
            update_file(mypath, force_i=1) #For media that is not fps boosted

    import my_gui_editor as mge
    mge_output_paths = [os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", filename) for filename in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW") if '.' in filename]
    mge_input_paths = [get_og_file_path(filename, force_i=1) for filename in mge_output_paths]
    mge_pre_call = update_file #Gets the output path of the media being saved, used to create history

    mge.temporary_file_location = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/temp.mp4"
    mge.launch(mge_input_paths, mge_output_paths, mge_pre_call)


    cube_library = [f"{os.path.basename(f)}" for f in glob.glob("luts/*.cube")]
    
    class FuzzyMusicCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text
            if text.strip():
                matches = process.extract(
                    text,
                    cube_library,
                    limit=10,  # number of suggestions
                    score_cutoff=40  # minimum similarity score
                )
                for match, score, _ in matches:
                    yield Completion(match.replace('<ansired>','').replace('</ansired>',''), start_position=-len(text), display=HTML(match))

    completer = FuzzyMusicCompleter()
    print("Pick a .cube file (defaults to no LUT): ")
    lut_name = prompt("LUT file: ", completer=completer)

    if len(lut_name.split(".")) >= 2:
        lut_interp = ffxcc.load_cube_lut(f"luts/{lut_name}")
        
        for filename in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW"):
            if filename.lower().endswith('.mp4'):
                video_path = os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", filename)

                input_filename = update_file(video_path) #Update history archive + get name of the latest archive

                cap = cv2.VideoCapture(input_filename)
                #Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_frame = ffxcc.apply_lut(frame, lut_interp)
                    out.write(out_frame)

                cap.release()
                out.release()


            
            elif filename.lower().endswith('.png'):
                img_path = os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", filename)

                input_filename = update_file(img_path) #Update history archive + get name of the latest archive

                img = cv2.imread(input_filename)
                out_img = ffxcc.apply_lut(img, lut_interp)
                cv2.imwrite(img_path, out_img) 

        print("Applied LUT...")            
                
    else:
        print("Ok no LUT...")     

## Managing RBs
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt"):
    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt") as file:
        redo_stuff = file.read()
    if redo_stuff.strip() == "0":
        redo_rb = False
    elif redo_stuff.strip() == "1":
        redo_rb = True
    else:
        raise Exception(f"Invalid redo.txt file @ {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt | delete if needed")
else:
    redo_rb = False

os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", exist_ok = True)
os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/TEMP", exist_ok = True)

number_of_media = len([f for f in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB") if os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", f))])

if number_of_media == 0 or redo_rb:
    input_folder = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW"
    output_folder = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB"
    temp_folder = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/TEMP"
    
    print("\nSeperating background from images...")
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + ".png")

            with open(input_path, "rb") as input_file:
                input_data = input_file.read()

            output_data = remove(input_data)

            with open(output_path, "wb") as output_file:
                output_file.write(output_data)
    
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            
            output_path1 = os.path.join(temp_folder, filename.rsplit('.', 1)[0] + "_front.png")
            output_path2 = os.path.join(temp_folder, filename.rsplit('.', 1)[0] + "_back.png")
            
            foutput_path1 = os.path.join(output_folder, filename.rsplit('.', 1)[0] + "_front.png")
            foutput_path2 = os.path.join(output_folder, filename.rsplit('.', 1)[0] + "_back.png")
            

            cap = cv2.VideoCapture(input_path)

            ret, frame = cap.read()
            cv2.imwrite(output_path1, frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            ret, frame = cap.read()
            cv2.imwrite(output_path2, frame)

            cap.release()


            with open(output_path1, "rb") as input_file:
                input_data = input_file.read()

            output_data = remove(input_data)

            with open(foutput_path1, "wb") as output_file:
                output_file.write(output_data)



            with open(output_path2, "rb") as input_file:
                input_data = input_file.read()

            output_data = remove(input_data)

            with open(foutput_path2, "wb") as output_file:
                output_file.write(output_data)
            

    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt", "w+") as file:
        file.write("0")

    print("All images foreground-background seperated! Delete the bad ones, then re-run this script.")
    print(f"Images are in: {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB/")
    exit()
else:
    print(f"Images already RBed, if you would like to redo it, delete all images in {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB or set {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/redo.txt to '1'")


print("Filters can be changed via the filters.json file. However you must restart the program to enact changes.")
if not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/filters.json"):
    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/filters.json", "w") as f:
        json.dump(FILTER_SETTINGS,f, indent=4)


### Create hook-video
def get_video_path(title):
    root = tk.Tk()
    root.withdraw()
    video_file_types = [
        ("Video files", "*.mp4 *.mov *.avi *.wmv *.mkv"),
        ("MP4 files", "*.mp4"),
        ("MOV files", "*.mov"),
        ("AVI files", "*.avi"),
        ("WMV files", "*.wmv"),
        ("MKV files", "*.mkv"),
        ("All files", "*.*")
    ]
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1) #Makes file explorer sharper
    except Exception as e:
        print(f"Could not set DPI awareness: {e}")


    file_path = filedialog.askopenfilename(title=title, filetypes=video_file_types)
    if os.path.exists(file_path):
        print(f"Selected: {file_path}")
    else:
        raise FileNotFoundError
    return file_path

if not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/nohook.txt"):
    if input("Would you like to add a hook/starting video? (Enter 'yes'): ").lower() not in {'y', 'yes'}:
        with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/nohook.txt", "w") as f:
            pass
    else:
        #Allows you to trim + crop if needed, then you can add it
        scene_pack_splitter.launch(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw2.mp4")

        input_filename = f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw2.mp4"
        output_filename = f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4"
        output_width,output_height = VIDEO_SIZE

        command = (
            f'ffmpeg -i "{input_filename}" '
            f'-vf "scale=w={output_width}:h={output_height}:force_original_aspect_ratio=increase,'
            f'crop=w={output_width}:h={output_height}" '
            f'{output_filename}'
        )

        os.system(command)
        os.remove(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw2.mp4")


if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav") and os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook.wav"):
    os.system(f'ffmpeg -i "{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook2.wav"')

    def get_duration(file_path):
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception(f"Error: Could not get duration for {file_path}. Make sure ffprobe is installed and the file exists.")


    print("Time to adjust music volume and hook volume.")
    print("You can enter both one by one, keeping it blank will leave it as is.")
    print("i'll play it after you click enter and you can decide whether to save and exit by typing 'exit'")
    print("For reference, the music vol = 1 for the rest of the edit")
    print("You can stop the preview at any time by pressing and holding 'q'")
    music_vol = 1
    hook_vol = 1
    music_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook.wav"
    hook_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook2.wav"
    duration_music = get_duration(music_path)
    duration_hook = get_duration(hook_path)


    
    while True:
        print("\nCurrent values: ")
        print(f"Music volume = {music_vol}")
        print(f"Hook volume = {hook_vol}")
        print("Enter new values (or exit to exit)")
        music_vol_inp = input("Enter the new music vol: ")
        if music_vol_inp.lower() == 'exit':
            break
        hook_vol_inp = input(" Enter the new hook vol: ")
        if hook_vol_inp.lower() == 'exit':
            break
        try:
            if music_vol_inp != "":
                music_vol = float(music_vol_inp)
            if hook_vol_inp != "":
                hook_vol = float(hook_vol_inp)
        except:
            pass 

        #os.system(f'ffmpeg -loglevel quiet -i "{music_path}" -i "{hook_path}" -filter_complex "[0:a]volume={music_vol}[a0];[1:a]volume={hook_vol}[a1];[a0][a1]amix=inputs=2:duration=longest" "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav"')
                
        filter_complex = ""
        if duration_music > duration_hook:
            trim_start_seconds = duration_music - duration_hook
            
            filter_complex = (
                f'"[0:a]volume={music_vol},atrim=start={trim_start_seconds}[a0];'
                f'[1:a]volume={hook_vol}[a1];'
                f'[a0][a1]amix=inputs=2:duration=shortest"'
            )
        else:
            delay_seconds = duration_hook - duration_music
            delay_ms = int(delay_seconds * 1000)
            fade_duration = max(min(2.5, duration_music - 0.5),0.1)
            
            filter_complex = (
                f'"[0:a]volume={music_vol},afade=t=in:st=0:d={fade_duration},adelay={delay_ms}|{delay_ms}[a0];'
                f'[1:a]volume={hook_vol}[a1];'
                f'[a0][a1]amix=inputs=2:duration=longest"'
            )
            
        final_command = (
            f'ffmpeg -y -loglevel quiet '
            f'-i "{music_path}" -i "{hook_path}" '
            f'-filter_complex {filter_complex} '
            f'"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav"'
        )
        
        print(final_command)
        os.system(final_command)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav")
        pygame.mixer.music.set_volume(1)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.25) # Wait a second and check again
            if keyboard.is_pressed('q'):
                break
        pygame.quit() #Cannot move this block outside or else file-lock wont be released

 
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook.wav"):
    os.system(f'ffmpeg -i "{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook2.wav"')

    def get_duration(file_path):
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception(f"Error: Could not get duration for {file_path}. Make sure ffprobe is installed and the file exists.")


    print("Time to adjust hook volume.")
    print("You can adjust the volume. You will here the hook play out then a portion of the music.")
    print("i'll play it after you click enter and you can decide whether to save and exit by typing 'exit'")
    print("You can stop the preview at any time by pressing and holding 'q'")
    hook_vol = 1
    music_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav"
    hook_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook2.wav"
    duration_music = get_duration(music_path)
    duration_hook = get_duration(hook_path)

    
    while True:
        print("\nCurrent values: ")
        print(f"Hook volume = {hook_vol}")
        print("Enter new values (or exit to exit)")
        hook_vol_inp = input("Enter the new hook vol: ")
        if hook_vol_inp.lower() == 'exit':
            break
        try:
            if hook_vol_inp != "":
                hook_vol = float(hook_vol_inp)
        except:
            pass 

        #os.system(f'ffmpeg -loglevel quiet -i "{music_path}" -i "{hook_path}" -filter_complex "[0:a]volume={music_vol}[a0];[1:a]volume={hook_vol}[a1];[a0][a1]amix=inputs=2:duration=longest" "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav"')
                

        delay_seconds = duration_hook
        delay_ms = int(delay_seconds * 1000)
        
        filter_complex = (
            f'"[0:a]volume={1},adelay={delay_ms}|{delay_ms}[a0];'
            f'[1:a]volume={hook_vol}[a1];'
            f'[a0][a1]amix=inputs=2:duration=longest"'
        )
            
        final_command = (
            f'ffmpeg -y -loglevel quiet '
            f'-i "{music_path}" -i "{hook_path}" '
            f'-filter_complex {filter_complex} '
            f'"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav"'
        )
        
        print(final_command)
        os.system(final_command)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav")
        pygame.mixer.music.set_volume(1)

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.25) # Wait a second and check again
            if keyboard.is_pressed('q'):
                break
        pygame.quit()#Cannot move this block outside or else file-lock wont be released

    os.system(f'ffmpeg -i "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook2.wav" -y -af "volume={hook_vol}" "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav"')


if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav") and os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro.wav"):
    os.system(f'ffmpeg -i "{PROJECT_FOLDER}/{PROJECT_NAME}/outro.mp4" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro2.wav"')

    def get_duration(file_path):
        """Gets the duration of a media file in seconds using ffprobe."""
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception(f"Error: Could not get duration for {file_path}. Make sure ffprobe is installed and the file exists.")


    print("Time to adjust music volume and outro volume.")
    print("You can enter both one by one, keeping it blank will leave it as is.")
    print("i'll play it after you click enter and you can decide whether to save and exit by typing 'exit'")
    print("For reference, the music vol = 1 for the rest of the edit")
    print("You can stop the preview at any time by pressing and holding 'q'")
    music_vol = 1
    hook_vol = 1
    music_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro.wav"
    hook_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro2.wav"
    duration_music = get_duration(music_path)
    duration_hook = get_duration(hook_path)


    
    while True:
        print("\nCurrent values: ")
        print(f"Music volume = {music_vol}")
        print(f"Outro volume = {hook_vol}")
        print("Enter new values (or exit to exit)")
        music_vol_inp = input("Enter the new music vol: ")
        if music_vol_inp.lower() == 'exit':
            break
        hook_vol_inp = input(" Enter the new outro vol: ")
        if hook_vol_inp.lower() == 'exit':
            break
        try:
            if music_vol_inp != "":
                music_vol = float(music_vol_inp)
            if hook_vol_inp != "":
                hook_vol = float(hook_vol_inp)
        except:
            pass 

                
        filter_complex = ""
        if duration_music > duration_hook:
            trim_start_seconds = 0 #Probably best to just remove the atrim filter
            
            filter_complex = (
                f'"[0:a]volume={music_vol},atrim=start={trim_start_seconds}[a0];'
                f'[1:a]volume={hook_vol}[a1];'
                f'[a0][a1]amix=inputs=2:duration=shortest"'
            )
        else:
            delay_seconds = 0
            delay_ms = int(delay_seconds * 1000)
            fade_duration = max(min(2.5, duration_music - 0.5),0.1)
            
            filter_complex = (
                f'"[0:a]volume={music_vol},afade=t=out:st={duration_music - fade_duration}:d={fade_duration},adelay={delay_ms}|{delay_ms}[a0];'
                f'[1:a]volume={hook_vol}[a1];'
                f'[a0][a1]amix=inputs=2:duration=longest"'
            )
            
        final_command = (
            f'ffmpeg -y -loglevel quiet '
            f'-i "{music_path}" -i "{hook_path}" '
            f'-filter_complex {filter_complex} '
            f'"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav"'
        )
        
        print(final_command)
        os.system(final_command)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav")
        pygame.mixer.music.set_volume(1)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.25) # Wait a second and check again
            if keyboard.is_pressed('q'):
                break
        pygame.quit()#Cannot move this block outside or else file-lock wont be released


 
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro.wav"):
    os.system(f'ffmpeg -i "{PROJECT_FOLDER}/{PROJECT_NAME}/outro.mp4" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro2.wav"')

    def get_duration(file_path):
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception(f"Error: Could not get duration for {file_path}. Make sure ffprobe is installed and the file exists.")


    print("Time to adjust outro volume.")
    print("You can adjust the volume. You will here the outro play out then a portion of the music.")
    print("i'll play it after you click enter and you can decide whether to save and exit by typing 'exit'")
    print("You can stop the preview at any time by pressing and holding 'q'")
    hook_vol = 1
    music_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav"
    hook_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro2.wav"
    duration_music = get_duration(music_path)
    duration_hook = get_duration(hook_path)

    
    while True:
        print("\nCurrent values: ")
        print(f"Outro volume = {hook_vol}")
        print("Enter new values (or exit to exit)")
        hook_vol_inp = input("Enter the new hook vol: ")
        if hook_vol_inp.lower() == 'exit':
            break
        try:
            if hook_vol_inp != "":
                hook_vol = float(hook_vol_inp)
        except:
            pass 

            

        delay_seconds = duration_hook
        delay_ms = int(delay_seconds * 1000)
        
        filter_complex = (
            f'"[0:a]volume={1},adelay={delay_ms}|{delay_ms}[a0];'
            f'[1:a]volume={hook_vol}[a1];'
            f'[a0][a1]amix=inputs=2:duration=longest"'
        )
            
        final_command = (
            f'ffmpeg -y -loglevel quiet '
            f'-i "{music_path}" -i "{hook_path}" '
            f'-filter_complex {filter_complex} '
            f'"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav"'
        )
        
        print(final_command)
        os.system(final_command)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav")
        pygame.mixer.music.set_volume(1)

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.25) # Wait a second and check again
            if keyboard.is_pressed('q'):
                break
        pygame.quit()#Cannot move this block outside or else file-lock wont be released



    os.system(f'ffmpeg -i "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro2.wav" -y -af "volume={hook_vol}" "{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav"')
 

#Add captions
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/caption.json"):
    caption.create_caption_json(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4", output_filename=f"{PROJECT_FOLDER}/{PROJECT_NAME}/caption.json")
    print('Captions created. You can edit them in caption.json, Deleting all entries (but maintaining the [] brackets, will remove captions.)')


import hashlib
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return str(h.hexdigest())

#Efficiency reasons. since this is quite time consuming, only creates the video if the .json has changed.
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4"):
    make_new_hook_vid = True

    if FILTER_SETTINGS["captions_on_hook"]:
        caption_hash = sha256_file(f"{PROJECT_FOLDER}/{PROJECT_NAME}/caption.json")
    else:
        caption_hash = "NO_CAPTIONS"
    
    if not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4"):
        make_new_hook_vid = True #if hook has been deleted/does not exist, we must make a new hook video       
    elif os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/caption_state.txt"):
        with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/caption_state.txt") as f:
            make_new_hook_vid = f.read() != caption_hash #Make new vid if they are different
    
    if make_new_hook_vid:
        print("Making hook video")       

    if FILTER_SETTINGS["captions_on_hook"] and make_new_hook_vid:
        caption.create_captions(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4", f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4", f"{PROJECT_FOLDER}/{PROJECT_NAME}/caption.json", additonal_filter=JUST_HOOK)
    elif make_new_hook_vid:
        cap = cv2.VideoCapture(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook_raw.mp4")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        #Define the codec and create a VideoWriter object.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4", fourcc, fps, (frame_width, frame_height))

        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()

            if not ret:
                break

            out.write(JUST_HOOK(frame))

        cap.release()
        out.release()
    else:
        print("Skipping creation of new hook video (since same as last time). Delete to regenerate.")

    with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/caption_state.txt", "w") as f:
        f.write(caption_hash) #set this up


# Create outro
if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro.mp4") and not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro_edited.mp4"):
    print("Creating outro")
    cap = cv2.VideoCapture(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro.mp4")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro_edited.mp4", fourcc, fps, VIDEO_SIZE)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(JUST_HOOK(cv2.resize(frame, VIDEO_SIZE)))

    cap.release()
    out.release()



print("Creating Edit!")

audio = AudioFileClip(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/song.wav")

beat_times = [float(x) for x in beat_times]
if beat_times[0] != 0:
    beat_times = [0] + beat_times
if beat_times[-1] != float(audio.duration):
    beat_times.append(float(audio.duration))

image_use_count = {f:0 for f in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW") if os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", f)) and f.lower().endswith('.png')}
video_use_count = {f:0 for f in os.listdir(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW") if os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW", f)) and f.lower().endswith('.mp4')}
prev = None
shown_rb = False


def get_optimal_media(typ = (True,True), requires_rb = False):
    global image_use_count
    global video_use_count
    global prev
    global shown_rb

    typ = ("i" if typ[0] else "") + ("v" if typ[1] else "")

    least_count = 100000

    if "i" in typ:
        for image in image_use_count.keys():
            if requires_rb and not os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", image)):
                if not shown_rb:
                    print(f"Couldn't find rb for image {image}")
                continue
            elif image == prev:
                continue
            else:
                least_count = min(image_use_count[image], least_count)

    if "v" in typ:
        for image in video_use_count.keys():
            f1_rb = os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", image.replace(".mp4", "_front.png")))
            f2_rb = os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", image.replace(".mp4", "_back.png")))

            if requires_rb and not (f1_rb and f2_rb):
                if not shown_rb:
                    print(f"Couldn't find rb for video {image}")
                continue
            elif image == prev:
                continue
            else:
                least_count = min(video_use_count[image], least_count)

    shown_rb = True
    potentials = []
    if "i" in typ:
        potentials += [f for f in image_use_count.keys() if image_use_count[f] == least_count and f!=prev]
    if "v" in typ:
        potentials += [f for f in video_use_count.keys() if video_use_count[f] == least_count and f!=prev]
    
    potentials2 = []
    if requires_rb:
        for x in potentials:
            if x.endswith(".png"):
                if os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", x)):
                    potentials2.append(x)
            elif x.endswith(".mp4"):
                f1_rb = os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", x.replace(".mp4", "_front.png")))
                f2_rb = os.path.isfile(os.path.join(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RB", x.replace(".mp4", "_back.png")))
                if f1_rb and f2_rb:
                    potentials2.append(x)
            else:
                raise Exception("Only .png and .mp4 are allowed.")

        potentials = potentials2
    
    random.seed()
    shuffle(potentials)
    for i in range(1000):
        for x in potentials:
            if x.endswith(f"P{i}.mp4"):
                potentials.remove(x)
                potentials = [x] + potentials
                break

    for i in range(1000, -1, -1):
        for x in potentials:
            if x.endswith(f"N{i}.mp4"):
                potentials.remove(x)
                potentials = potentials + [x]
                break
    

    for item in potentials:
        if item in settings.top_files:
            potentials = [item] + potentials
            break

    if potentials == []:
        return None
    try:
        image_use_count[potentials[0]] += 1
    except:
        video_use_count[potentials[0]] += 1

    prev = potentials[0]
    return f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW/{potentials[0]}"




if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/linear_sequence.json") and not ALWAYS_MAKE_NEW_LS:
    linear_sequence = json_writer.read(f"{PROJECT_FOLDER}/{PROJECT_NAME}/linear_sequence.json", json_columns)
    print("Using prior linear sequence -> delete linear_sequence.json for a new one...\n")
else:

    if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/template.pkl"):
        with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/template.pkl", "rb") as f:
            linear_sequence = pickle.load(f)

    else:
        #Contains timeline information
        #Each one refers to a singular media
        # Start of block, end of in-transition, start of out transition, end of block, transition function (inward), transition function (outward), bpm, filepath for block, beats_list (for inward transition)

        linear_sequence = [[beat_times[i], beat_times[i] + ((beat_times[i+1] - beat_times[i])*block_splits[0]), beat_times[i] + ((beat_times[i+1] - beat_times[i])*block_splits[1]), beat_times[i+1], None, None, 60/(beat_times[i+1] - beat_times[i]), None, None] for i in range(len(beat_times) - 1)]

        def calc_bpm(start,end):
            return 60/abs(start-end)


        if not os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/settings.json"):
            with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/settings.json", "w+") as f:
                f.write(f"""{{
            "NAME": "{PROJECT_NAME}"
        }}""")

        RESET_TRANS = False
        try:
            with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/transition_edit.json", 'r') as f:
                x = f.read().replace('False', 'false').replace('True','true')

            with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/transition_edit.json", 'w+') as f:
                f.write("\n".join([re.sub(r"//.*", "", line) for line in x.split("\n")])) #Remove comments

            x = "\n".join([i[2:] for i in x.split("\n") if i.startswith("//")])
            if "$RESET$" in x:
                RESET_TRANS = True
                x = x.replace("$RESET$", "")

            x = x.replace("$PROJECT$", PROJECT_NAME)
            x = x.replace("$SETTINGS$", f"{PROJECT_FOLDER}/{PROJECT_NAME}/settings.json")
            if x != "":
                print(x)
                print("Make the changes!")
                input("Click 'Enter' to continue...")

            with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/transition_edit.json", 'r') as f:
                ttransition_edits = json.load(f)
            
            if RESET_TRANS:
                for key in TRANSITIONS:
                    TRANSITIONS[key]["weight"] = 0

            for key in ttransition_edits:
                for key2 in ttransition_edits[key]:
                    TRANSITIONS[key][key2] = ttransition_edits[key][key2]
        except FileNotFoundError:
            pass


        with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/settings.json", 'r') as f:
            TRANSITION_SETTINGS = json.load(f)

        transition_registry.transitions.TRANSITION_SETTINGS = TRANSITION_SETTINGS

        #First assign the last/first loop transition
        potentials = []
        for key in TRANSITIONS.keys():
            #if not (TRANSITIONS[key]["lowest_bpm"] <= linear_sequence[-1][6] <= TRANSITIONS[key]["highest_bpm"]):
            if not (TRANSITIONS[key]["lowest_bpm"] <= calc_bpm(linear_sequence[-1][2], linear_sequence[-1][3]) <= TRANSITIONS[key]["highest_bpm"]):
                continue
            
            flag = False
            for extra_ind in range(TRANSITIONS[key]["beats"]):
                #if not (TRANSITIONS[key]["lowest_bpm"] <= linear_sequence[extra_ind][6] <= TRANSITIONS[key]["highest_bpm"]):
                if extra_ind == TRANSITIONS[key]["beats"] - 1: #Last iteration
                    calced_bpm = calc_bpm(linear_sequence[extra_ind][0], linear_sequence[extra_ind][1])
                else:
                    calced_bpm = calc_bpm(linear_sequence[extra_ind][0], linear_sequence[extra_ind][3])
                if not (TRANSITIONS[key]["lowest_bpm"] <= calced_bpm <= TRANSITIONS[key]["highest_bpm"]):
                    flag = True
                    break
            if flag:
                continue

            if TRANSITIONS[key]["intro_priority"]:
                weight = (TRANSITIONS[key]["weight"]+1) * 1e9
            else:
                weight = TRANSITIONS[key]["weight"]*1e-6

            potentials.append((key,weight))

        if potentials == []:
            raise Exception("Wierd BPMs no transitions loaded for first bit.")

        potentials = [(ab,max(0,bc)) for ab,bc in potentials]
        choices, weights = zip(*potentials)

        random.seed()
        my_transition = random.choices(choices, weights, k=1)[0]
        linear_sequence[-1][5] = str(my_transition)

        beats = TRANSITIONS[my_transition]["beats"]
        linear_sequence[beats-1][4] = my_transition
        linear_sequence[beats-1][8] = [float(linear_sequence[i][0]) for i in range(beats)]

        linear_sequence[beats-1][0] = linear_sequence[0][0]

        if beats != 1:
            for i in range(beats-1):
                linear_sequence[i] = None

        linear_sequence = [x for x in linear_sequence if x is not None]

        #Assign transitions
        prev_transitions = [str(my_transition)]

        for i in range(len(linear_sequence) - 1): #Assigning remaining transitions (outward to i, inward to i+1)
            if linear_sequence[i] == None:
                continue
            potentials = []
            for key in TRANSITIONS.keys():
                if TRANSITIONS[key]["beats"] > len(linear_sequence) - 1 - i: #Not enough beats left
                    continue

                #if not (TRANSITIONS[key]["lowest_bpm"] <= linear_sequence[i][6] <= TRANSITIONS[key]["highest_bpm"]):
                if not (TRANSITIONS[key]["lowest_bpm"] <= calc_bpm(linear_sequence[i][2], linear_sequence[i][3]) <= TRANSITIONS[key]["highest_bpm"]):
                    continue

                
                flag = False
                for extra_ind in range(TRANSITIONS[key]["beats"]):
                    if extra_ind == TRANSITIONS[key]["beats"] - 1: #Last iteration
                        calced_bpm = calc_bpm(linear_sequence[i + 1 + extra_ind][0], linear_sequence[i + 1 + extra_ind][1])
                    else:
                        calced_bpm = calc_bpm(linear_sequence[i + 1 + extra_ind][0], linear_sequence[i + 1 + extra_ind][3])
                    
                    #if not (TRANSITIONS[key]["lowest_bpm"] <= linear_sequence[i + 1 + extra_ind][6] <= TRANSITIONS[key]["highest_bpm"]):
                    if not (TRANSITIONS[key]["lowest_bpm"] <= calced_bpm <= TRANSITIONS[key]["highest_bpm"]):
                        flag = True
                        break
                if flag:
                    continue

                weight = float(TRANSITIONS[key]["weight"])
                if prev_transitions[-1] in TRANSITIONS[key]["boosts"]:
                    weight += TRANSITIONS[key]["boosts"][prev_transitions[-1]]

                for prev_tran in prev_transitions[-7:-6]:
                    if prev_tran in TRANSITIONS[key]["boosts_long_term"]:
                        weight += TRANSITIONS[key]["boosts_long_term"][prev_tran]


                potentials.append((key,weight))

            potentials = [(ab,max(0,bc)) for ab,bc in potentials]
            choices, weights = zip(*potentials)
            random.seed()
            my_transition = random.choices(choices, weights, k=1)[0]
            prev_transitions.append(my_transition)

            linear_sequence[i][5] = str(my_transition)
            beats = TRANSITIONS[my_transition]["beats"]
            linear_sequence[i+beats][4] = str(my_transition)

            linear_sequence[i+beats][8] = [float(linear_sequence[j + i + 1][0]) for j in range(beats)]
            linear_sequence[i+beats][0] = linear_sequence[i+1][0]

            if beats != 1:
                for j in range(beats-1):
                    linear_sequence[i + j + 1] = None


        linear_sequence = [x for x in linear_sequence if x is not None]


    #Assign media
    for i in range(len(linear_sequence)):
        a = None

        t1, t2 = linear_sequence[i][4], linear_sequence[i][5]
        t1, t2 = TRANSITIONS[t1], TRANSITIONS[t2]

        requires_rb = False
        prefers_video = False

        if "requires_rb" in t1:
            requires_rb = t1["requires_rb"]
        if "requires_rb_out" in t1:
            requires_rb = t1["requires_rb_out"] or requires_rb #Only 1 needed
        if "requires_rb" in t2: #Does not already require rb
            requires_rb = t2["requires_rb"] or requires_rb
        if "requires_rb_in" in t2: #Does not already require rb
            requires_rb = t2["requires_rb_in"] or requires_rb
        


        if "prefers_video" in t1:
            prefers_video = t1["prefers_video"]

        if "prefers_video" in t2:
            prefers_video = t2["prefers_video"] or prefers_video

        
        if prefers_video: 
            a = get_optimal_media((False, True), requires_rb)
        if a is None:
            a = get_optimal_media((True, True), requires_rb)
        if a is None:
            raise Exception("BRO WTF")

        linear_sequence[i][7] = a
        linear_sequence[i][0] = float(linear_sequence[i][0])
        linear_sequence[i][1] = float(linear_sequence[i][1])
        linear_sequence[i][2] = float(linear_sequence[i][2])
        linear_sequence[i][3] = float(linear_sequence[i][3])
        
    #Video-snapping
    if ENABLE_VIDEO_SNAPPING:
        if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/vid_snap.txt"):
            with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/vid_snap.txt", "r") as f:
                files_to_snap = [x.strip() for x in f.read().split("\n")]
        else:
            with open(f"{PROJECT_FOLDER}/{PROJECT_NAME}/vid_snap.txt", "w") as f:
                f.write("all")
            print("Write the filenames in vid_snap.txt to snap. It will be snapped. Write 'all' for everything")
            files_to_snap = ["all"]

        get_duration_cache = {}
        def get_duration(file_path):
            global get_duration_cache
            if file_path in get_duration_cache:
                return get_duration_cache[file_path]
            
            command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                get_duration_cache[file_path] = float(result.stdout)
                return float(result.stdout)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise Exception(f"Error: Could not get duration for {file_path}. Make sure ffprobe is installed and the file exists.")

        linear_sequence2 = []
        was_just_snapped = False
        for i in range(len(linear_sequence)):
            if len(linear_sequence2) > 0:
                if_updated_duration = linear_sequence[i][2] - linear_sequence2[-1][1] #Video duration if 'snapped'

            #if the last one on linear_sequence2 is a video AND the updated duration is still less than the video's duration
            if len(linear_sequence2) > 0 and (any([(filename in linear_sequence2[-1][7]) for filename in files_to_snap]) or ('all' in files_to_snap)) and linear_sequence2[-1][7].endswith(".mp4") and if_updated_duration <= get_duration(linear_sequence2[-1][7]):
                print(f"Snapped: {linear_sequence2[-1][7]}")
                linear_sequence2[-1][2] = linear_sequence[i][2]
                linear_sequence2[-1][3] = linear_sequence[i][3]
                linear_sequence2[-1][5] = "default" #Set it to None for now, set the new transition later
                was_just_snapped = True
                if settings.LS_REBALANCER:
                    print(f"Attempted to rebalance {linear_sequence[i][7]}")
                    print([os.path.basename(x[7]) for x in linear_sequence])
                    
                    prev_item = str(linear_sequence[i][7]) #Push media ahead one step
                    try:
                        for j in range(1, len(linear_sequence)):
                            prev_item2 = str(linear_sequence[i + j][7])
                            linear_sequence[i + j][7] = str(prev_item)
                            prev_item = str(prev_item2)
                    except IndexError:
                        pass
                    print([os.path.basename(x[7]) for x in linear_sequence])
                    

            else:
                linear_sequence2.append(linear_sequence[i]) #not applicable to be snapped, move on
                if was_just_snapped:
                    was_just_snapped = False
                    #Now need to check if the current inward transition is compatible
                    #rb_path1 = linear_sequence2[-2][7].replace('/RAW/','/RB/').replace(".mp4", "_front.png")
                    rb_path2 = linear_sequence2[-2][7].replace('/RAW/','/RB/').replace(".mp4", "_back.png")
                    
                    if not (TRANSITIONS[linear_sequence2[-1][4]]['requires_rb_in'] or TRANSITIONS[linear_sequence2[-1][4]]['requires_rb']) or os.path.exists(rb_path2):
                        linear_sequence2[-2][5] = linear_sequence2[-1][4] #It is compatible and now set up
                    else: #Incompatible
                        print(f"Snapping video {linear_sequence2[-2][7]} led to incompatible transition. Attempting to resolve...")
                        
                        potentials = [key for key in TRANSITIONS.keys() if not (TRANSITIONS[key]['requires_rb_in'] or TRANSITIONS[key]['requires_rb'])] #Does not need rb-in
                        potentials2 = [key for key in potentials if len(linear_sequence2[-1][8]) == TRANSITIONS[key]['beats']]
                        if len(potentials2) == 0:
                            print("Exact match not found. Attempting syntethic beat generation.")
                            #Get heavily weighted weights (based on inv. of dist. between synth beats and actual beats)
                            wts = [TRANSITIONS[key]["weight"]/((TRANSITIONS[key]['beats']-len(linear_sequence2[-1][8]))**4) for key in potentials]
                            random.seed()
                            chosen = linear_sequence2[-2][5] = linear_sequence2[-1][4] = random.choices(potentials, wts)[0]

                            #To create synthetic weights
                            start_time = linear_sequence2[-2][2]
                            end_time = linear_sequence2[-1][1]
                            del_t = (end_time - start_time)/(TRANSITIONS[chosen]["beats"]+1) #Step size
                            
                            #Create beats
                            linear_sequence2[-1][8] = [start_time + (ik*del_t) for ik in range(1, 1 + TRANSITIONS[chosen]["beats"])]
                        else:
                            print("Exact alternative found.")
                            wts = [TRANSITIONS[key]["weight"] for key in potentials2]
                            random.seed()

                            linear_sequence2[-2][5] = linear_sequence2[-1][4] = random.choices(potentials2, wts)[0]


        linear_sequence = list(linear_sequence2)

    #Check if irregularities:
    for a in linear_sequence:
        if any([i is None for i in a]):
            raise Exception("Bruh something wrong ;-;")

    if TOTAL_TRANSITION_TIME:
        for i in range(len(linear_sequence)):
            linear_sequence[i][1] = linear_sequence[i][2] = (linear_sequence[i][1] + linear_sequence[i][2])/2

    #Linear Sequence created!!
    json_writer.write(linear_sequence, f"{PROJECT_FOLDER}/{PROJECT_NAME}/linear_sequence.json", columns=json_columns)



#Adds hook
if ENABLE_HOOK_INJECTION and os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4"):
    try_to_seperate = True
    try:
        if not make_new_hook_vid:
            try_to_seperate = False
    except:
        pass

    if try_to_seperate:
        #Start by creating a integrated_hook
        os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK", exist_ok=True)
        os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RAW", exist_ok=True)
        os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB", exist_ok=True)
        os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB_HIGHLIGHT", exist_ok=True)
            

        print("\nSeperating background...")
        input_path = f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4"
        output_path1 = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/hook{1}.png"
        output_path2 = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/hook{2}.png"
        temp1 = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RAW/hook{1}.png"
        temp2 = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RAW/hook{2}.png"
        
        cap = cv2.VideoCapture(input_path)

        ret, frame = cap.read()
        cv2.imwrite(temp1, frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        ret, frame = cap.read()
        cv2.imwrite(temp2, frame)

        cap.release()


        with open(temp1, "rb") as input_file:
            input_data = input_file.read()
        output_data = remove(input_data)
        with open(output_path1, "wb") as output_file:
            output_file.write(output_data)

        with open(temp2, "rb") as input_file:
            input_data = input_file.read()
        output_data = remove(input_data)
        with open(output_path2, "wb") as output_file:
            output_file.write(output_data)
        
        print(f"Please check {PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/ for the RBed images. Delete the ones that don't look")
        input("Click enter twice to continue...")
        input("Click enter again to continue...")
    
    else:
        print("Skipping hrb...")
    
    rb_approved1 = os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/hook{1}.png") and os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/hook{1}.png")
    rb_approved2 = os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/hook{2}.png") and os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RB/hook{2}.png")
    
    linear_sequence[0][7] = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RAW/hook{2}.png"
    linear_sequence[-1][7] = f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/HOOK/RAW/hook{1}.png"

    
    #if this causes an issue again, make this safer, require entire transition to NOT require rb
    if (not rb_approved1) or (not rb_approved2):
        if TRANSITIONS[linear_sequence[0][4]]['requires_rb'] or TRANSITIONS[linear_sequence[0][4]]['requires_rb_in'] or TRANSITIONS[linear_sequence[0][4]]['requires_rb_out']: #Inward transition
            linear_sequence[-1][5] = linear_sequence[0][4] = "default"
            linear_sequence[0][8] = [linear_sequence[0][0]]        
        if TRANSITIONS[linear_sequence[0][5]]['requires_rb'] or TRANSITIONS[linear_sequence[0][4]]['requires_rb_out'] or TRANSITIONS[linear_sequence[0][4]]['requires_rb_in']: #Outward transition
            linear_sequence[0][5] = linear_sequence[1][4] = "default"
            linear_sequence[1][8] = [linear_sequence[1][0]]

        if TRANSITIONS[linear_sequence[-1][4]]['requires_rb'] or TRANSITIONS[linear_sequence[-1][4]]['requires_rb_in'] or TRANSITIONS[linear_sequence[-1][4]]['requires_rb_out']: #Inward transition
            linear_sequence[-2][5] = linear_sequence[-1][4] = "default"
            linear_sequence[-1][8] = [linear_sequence[-1][0]]
        if TRANSITIONS[linear_sequence[-1][5]]['requires_rb'] or TRANSITIONS[linear_sequence[-1][4]]['requires_rb_out'] or TRANSITIONS[linear_sequence[-1][4]]['requires_rb_in']: #Outward transition
            linear_sequence[-1][5] = linear_sequence[0][4] = "default"
            linear_sequence[0][8] = [linear_sequence[0][0]]
    

print("\n\n\nLinear Sequence (Timeline):\n")
pretty_table = [["Start", "End in-t", "Start o-t", "End", "in-tran", "o-tran", "filepath", "beats (in-t)"]]
for item in linear_sequence:
    #Basically get every item but after some edits (shortenting numbers, removing unnecessary file)
    r = [round(x,2) if type(x) in {float, int} else ([round(j,2) if type(j) in {float, int} else j for j in x] if type(x) == list else (x.replace(f"{PROJECT_FOLDER}/{PROJECT_NAME}/MEDIA/RAW/", "") if type(x) == str else x)) for x in item]
    a,b,c,d,e,f,_,h,i = r
    pretty_table.append([a,b,c,d,e,f,h,i]) #Remove bpm


pretty_table = [[str(i) for i in x] for x in pretty_table]
col_widths = [max(len(str(row[i])) for row in pretty_table) for i in range(len(pretty_table[0]))]

print("\033[31m" + " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(pretty_table[0])).replace("|", "\033[0m|\033[31m") + "\033[0m")
for row in pretty_table[1:]:
    print(" | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)))


def get_block_ind(t):
    for ind,x in enumerate(linear_sequence):
        start,_,_,end,_,_,_,_,_ = x
        if start <= t <= end:
            return ind


image_cache = {}
def load(image_path, video_size = VIDEO_SIZE):
    if image_path in image_cache:
        return image_cache[image_path]

    target_w, target_h = video_size

    #Loads image as BGR
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    h, w = img.shape[:2]


    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]

    image_cache[image_path] = cropped
    return cropped

video_cache = {}
def vid_load(video_path, progress, video_size = VIDEO_SIZE):
    if video_path in video_cache:
        cap, total_frames = video_cache[video_path]
    else:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        video_cache[video_path] = cap, total_frames

    frame_number = int(total_frames * progress)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, img = cap.read()
    if not ret:
        raise Exception("Couldn't read!")


    target_w, target_h = video_size
    h, w = img.shape[:2]

    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]
    return cropped


def transition_wrapper(transition, f1,f2,p,beats,total_time, global_block_progress):
    if DEBUG_MODE:
        return transition(f1,f2,p,beats,total_time, global_block_progress)
    try:
        return transition(f1,f2,p,beats,total_time, global_block_progress)
    except TypeError:
        pass
    try:
        return transition(f1,f2,p,beats,total_time)
    except TypeError:
        pass
    try:
        return transition(f1,f2,p,beats)
    except TypeError:
        return transition(f1,f2,p)


def expand_linear_sequence(linear_sequence_raw):
    end = list(linear_sequence_raw[-1])
    end[1] -= end[0]
    end[2] -= end[0]
    end[3] -= end[0]
    end[0] -= end[0]
    
    addi = end[3] #How much to add to each val

    bichme_ka = [[a + addi,b + addi,c + addi,d + addi,e,f,g,h,i] for a,b,c,d,e,f,g,h,i in linear_sequence_raw] #Adds additional value

    front = list(linear_sequence_raw[0])
    front[0] += bichme_ka[-1][3]
    front[1] += bichme_ka[-1][3]
    front[2] += bichme_ka[-1][3]
    front[3] += bichme_ka[-1][3]

    return [end] + bichme_ka + [front], addi #Create new looped linear sequence + offset time


def get_block_progresses_single(linear_sequence_raw, my_ind_raw, t_raw):
    l1, addi1 = expand_linear_sequence(linear_sequence_raw)
    linear_sequence, addi2 = expand_linear_sequence(l1)

    my_ind = my_ind_raw + 2
    t = t_raw + addi1 + addi2
        
    current_block = linear_sequence[my_ind]
    
    prior_block = linear_sequence[my_ind - 1]

    next_block = linear_sequence[my_ind + 1]

    #Get current block block-progress
    current_block_start, _, _, current_block_end, tfi_name, tfo_name, _, _, _ = list(current_block)
    
    extend_in = TRANSITIONS[tfi_name].get("extend_block_progress", False) #Allows function to determine whether to extend the block progress of both
    extend_out = TRANSITIONS[tfo_name].get("extend_block_progress", False)
    
    requires_rb_tfi = TRANSITIONS[tfi_name].get("requires_rb", False) or TRANSITIONS[tfi_name].get("requires_rb_out", False)
    requires_rb_tfo = TRANSITIONS[tfo_name].get("requires_rb", False) or TRANSITIONS[tfo_name].get("requires_rb_in", False)

    if extend_in:
        current_block_start = prior_block[2] #Make current block start -> prior block s2
    if extend_out:
        current_block_end = next_block[1] #Make current block end -> prior next block e1
    
    if requires_rb_tfi:
        current_block_start = current_block[1]
    if requires_rb_tfo:
        current_block_end = current_block[2]

    if current_block_start == current_block_end:
        CURRENT_BLOCK_PROGRESS = 0 if t < current_block_start else 1    
    else:
        CURRENT_BLOCK_PROGRESS = (t - current_block_start)/(current_block_end - current_block_start) #Remains the same as always, IF both extend_in and extend_out are False

    CURRENT_BLOCK_PROGRESS = max(0, CURRENT_BLOCK_PROGRESS)
    CURRENT_BLOCK_PROGRESS = min(1, CURRENT_BLOCK_PROGRESS)

    return CURRENT_BLOCK_PROGRESS


def get_block_progress(linear_sequence, my_ind, t):
    return get_block_progresses_single(linear_sequence, my_ind - 1, t), get_block_progresses_single(linear_sequence, my_ind, t), get_block_progresses_single(linear_sequence, my_ind + 1, t)


def my_frame_function(t):
    """
    Common problems:
     - block_progress is an integer/tuple -> check if your rb settings are correct in transition_registry
    """
    my_ind = get_block_ind(t)
    s1,e1,s2,e2, tfi_name, tfo_name, bpm, filepath, _ = linear_sequence[my_ind]

    tfi = TRANSITIONS[tfi_name]["func"]
    tfo = TRANSITIONS[tfo_name]["func"]
    
    block_progress = global_block_progress = (t-s1)/(e2-s1)

    #Only matters for block_progress
    requires_rb_tfi = TRANSITIONS[tfi_name].get("requires_rb", False) or TRANSITIONS[tfi_name].get("requires_rb_out", False)
    requires_rb_tfo = TRANSITIONS[tfo_name].get("requires_rb", False) or TRANSITIONS[tfo_name].get("requires_rb_in", False)

    ask_for_complete_tfi = TRANSITIONS[tfi_name].get("requires_rb_out", False)
    ask_for_complete_tfo = TRANSITIONS[tfo_name].get("requires_rb_in", False)

    new_block_progress1, new_block_progress2, new_block_progress3 = get_block_progress(linear_sequence, my_ind, t) #Prev, current, next block progresses

    if t <= e1 and requires_rb_tfi and ask_for_complete_tfi:
        global_block_progress = block_progress = (new_block_progress1, 0)
    elif t >= s2 and requires_rb_tfo and ask_for_complete_tfo:
        global_block_progress = block_progress = (1, new_block_progress3)
    elif t <= e1 and requires_rb_tfi:
        global_block_progress = block_progress = 0
    elif t >= s2 and requires_rb_tfo:
        global_block_progress = block_progress = 1
    else:
        if t <= e1:
            global_block_progress = block_progress = (new_block_progress1, new_block_progress2)
        elif t >= s2:
            global_block_progress = block_progress = (new_block_progress2, new_block_progress3)
        else:
            global_block_progress = block_progress = new_block_progress2

    #Special cases:
    if t <= e1 and my_ind == 0:
        prior_block = linear_sequence[-1]
        total_transition_time = (e1 - s1) + (prior_block[3] - prior_block[2])
        transition_progress = t + (prior_block[3] - prior_block[2])
        transition_progress = transition_progress/total_transition_time

        #Convert beat times into progress time
        beat_times = [x + (prior_block[3] - prior_block[2]) for x in linear_sequence[my_ind][8]]
        beat_times = [x/total_transition_time for x in beat_times]

        return transition_wrapper(tfi, prior_block[7], filepath, transition_progress, beat_times, total_transition_time, global_block_progress)
        return tfi(prior_block[7], filepath, transition_progress)

    if t >= s2 and my_ind == len(linear_sequence) - 1:
        next_block = linear_sequence[0]
        total_transition_time = (e2 - s2) + (next_block[1] - next_block[0])
        transition_progress = t - s2
        transition_progress = transition_progress/total_transition_time

        beat_times = [x + (e2-s2) for x in next_block[8]]
        beat_times = [x/total_transition_time for x in beat_times] #Technically same beat_times as for my_ind = 0

        return transition_wrapper(tfo, filepath, next_block[7], transition_progress, beat_times, total_transition_time, global_block_progress)
        return tfo(filepath, next_block[7], transition_progress)


    if e1 < t < s2:
        if filepath.endswith(".png"):
            return load(filepath)
        elif filepath.endswith(".mp4"):
            return vid_load(filepath, block_progress)
    elif t <= e1:
        prior_block = linear_sequence[my_ind - 1]
        transition_progress = (t - prior_block[2])/(e1 - prior_block[2])

        beat_times = [(x - prior_block[2])/(e1 - prior_block[2]) for x in linear_sequence[my_ind][8]]

        return transition_wrapper(tfi, prior_block[7] ,filepath, transition_progress, beat_times, (e1 - prior_block[2]), global_block_progress)
        return tfi(prior_block[7] ,filepath, transition_progress)
    elif t >= s2:
        next_block = linear_sequence[my_ind + 1]
        transition_progress = (t - s2)/(next_block[1] - s2)

        beat_times = [(x - s2)/(next_block[1] - s2) for x in next_block[8]]

        return transition_wrapper(tfo, filepath, next_block[7], transition_progress, beat_times, (next_block[1] - s2), global_block_progress)
        return tfo(filepath, next_block[7], transition_progress)
    

    raise Exception("BROSKEY!!!")



def frame_rgb(t):
    frame_bgr = my_frame_function(t)
    
    #Apply cc
    #frame_bgr = ffxcc.cc_main(frame_bgr)

    #Apply filters
    #frame_bgr = ffxcc.light_vhs_filter(frame_bgr, strength = 1, FPS = FPS)
    frame_bgr = JUST_EDIT(frame_bgr)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


THREADING = False
THREADS = 6
DEBUG_MODE2 = False #To allow multiple renders at once

#Not very well tested, is prone to breaking. However CAN boost speed quite a bit (3-5x) when enabled
if THREADING:
    from moviepy import VideoFileClip
    import multiprocessing as mp
    
    if DEBUG_MODE2:
        idd = random.randint(1_000_000, 10_000_000 - 1)
        print(f"Saving as {PROJECT_NAME}{idd}.mp4\n\n\n")
    
    os.makedirs(f"{PROJECT_FOLDER}/{PROJECT_NAME}/RENDER{idd}/", exist_ok= True)
    def make_chunk(frame_func, start, end, fps):
        # Generate a clip for this segment
        return VideoClip(lambda t: frame_func(t + start), duration=end - start).set_fps(fps)

    def render_chunk(args):
        frame_func, start, end, fps, idx = args
        clip = make_chunk(frame_func, start, end, fps)
        filename = f"{PROJECT_FOLDER}/{PROJECT_NAME}/RENDER{idd}/chunk_{idx}.mp4"
        clip.write_videofile(filename, fps=fps, codec="libx264", audio=False, logger=None)
        return filename

    def parallel_render(frame_func, duration, fps, n_chunks, output_file):
        # Split into chunks
        chunk_length = duration / n_chunks
        jobs = []
        for i in range(n_chunks):
            start = i * chunk_length
            end = (i+1) * chunk_length if i < n_chunks-1 else duration
            jobs.append((frame_func, start, end, fps, i))

        # Run in parallel
        with mp.Pool(n_chunks) as pool:
            filenames = pool.map(render_chunk, jobs)

        # Concatenate results
        clips = [VideoFileClip(fn) for fn in filenames]
        final = concatenate_videoclips(clips)
        final.write_videofile(output_file, fps=fps, codec="libx264", audio_codec="aac")
    
    
    if DEBUG_MODE2:
        idd = random.randint(1_000_000, 10_000_000 - 1)
        parallel_render(frame_rgb,audio.duration/2, FPS, THREADS, f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}{idd}.mp4")
    else:
        parallel_render(frame_rgb,audio.duration, FPS, THREADS, f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}.mp4")
    
    exit()


#No threading
clip = VideoClip(frame_rgb, duration=audio.duration)

clip = clip.with_audio(audio)




if DEBUG_MODE2:
    idd = random.randint(1_000_000, 10_000_000 - 1)
    print(f"Saving as {PROJECT_NAME}{idd}.mp4\n\n\n")
    clip.write_videofile(f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}{idd}.mp4", fps=FPS, codec="libx264", audio_codec="aac")
else:
    #Comment out if needed
    clip.write_videofile(f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}.mp4", fps=FPS, codec="libx264", audio_codec="aac")
    
    del clip
    clip = VideoFileClip(f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}.mp4")
    if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4"):
        hook_clip = VideoFileClip(f"{PROJECT_FOLDER}/{PROJECT_NAME}/hook.mp4")
        hook_audio = AudioFileClip(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/hook3.wav")
        hook_clip = hook_clip.with_audio(hook_audio)
    else:
        hook_clip = None

    if os.path.exists(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro_edited.mp4"):
        outro_clip = VideoFileClip(f"{PROJECT_FOLDER}/{PROJECT_NAME}/outro_edited.mp4")
        outro_audio = AudioFileClip(f"{PROJECT_FOLDER}/{PROJECT_NAME}/SONG/outro3.wav")

        outro_clip = outro_clip.with_audio(outro_audio.with_effects([AudioFadeOut(outro_clip.duration)]))
        outro_clip.write_videofile(f"{PROJECT_FOLDER}/{PROJECT_NAME}/debug.mp4", ffmpeg_params=["-pix_fmt", "yuv420p"], fps=FPS, codec="libx264", audio_codec="aac")
    else:
        outro_clip = None

    consolidated = [hook_clip, clip, outro_clip]
    output_hook = concatenate_videoclips([cl for cl in consolidated if cl is not None])
    output_hook.write_videofile(f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}_consolidated.mp4", fps=FPS, codec="libx264", audio_codec="aac")

    # Un comment for looped preview
    #looped_preview = concatenate_videoclips([cl for cl in (consolidated+consolidated) if cl is not None])
    #looped_preview.write_videofile(f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}_loop_preview.mp4", fps=FPS, codec="libx264", audio_codec="aac")

for key in video_cache.keys():
    video_cache[key][0].release()

#time.sleep(2)
#os.startfile(f"{PROJECT_FOLDER}/{PROJECT_NAME}/{PROJECT_NAME}_loop_preview.mp4")

print("TIP: You can add outro.mp4 and watermark.png if you haven't already")