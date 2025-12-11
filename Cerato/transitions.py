#Each transition is just a function -> log them in transition registry to activate
#Every transition should be able to handle image/video -> image/video
#Use the linear_cut to get initial frame if you want -> it can automatically use image/video, and you can manipulate progress if need be
#since progress from p = total progress of transition however if multiple beats from beats -> you may want to re-factor the linear cut to only change on last beat

### Some transitions which were not added yet, and still need to be built upon
# colour_cut

"""
Creating new transition:
transition(f1,f2,p, beats,total_time) #beats and total_time are optional, but are positional, so if you take total_time you must take beats

here f1 and f2 are the *filepaths* of the realted media
p = [0,1] -> current progress throughout the transition
beats = [list of beats] -> each beat = [0,1] -> when the music's beat is there
total_time = total_time if you need tru time conversion

output is a cv2 BGR frame

Also make sure to add in comments in the transition itself which helps create the transition registry for itself/other LINKED transitions. Especially for top level '_transition_templates'

Log it in transition_registry to activate
"""

from settings import VIDEO_SIZE, ENABLE_TRANSITION_PROMPTS
import cv2
from transformers import *
import progress_func
import math
import random
import os
import transformers
import hashlib
import rembg_video

TRANSITION_SETTINGS = {}

# Transitoin template (basic)
def transition(f1,f2,p, beats, total_time, block_progress):
    """
    f1: media 1 
    f2: media 2
    p: [0,1] determines how much percentage is done
    beats: [list of [0,1]] -> beats requested
    total_time: total time in seconds of the TRANSITION
    block_progress = ([0,1], [0,1]) -> progress of f1,f2

    must return a single cv2 frame (BGR)
    """
    pass
    # return frame # -> frame = BGR cv2 frame


image_cache = {}
def load(image_path, video_size = VIDEO_SIZE, alpha = False):
    if image_path in image_cache:
        return image_cache[image_path].copy()

    target_w, target_h = video_size

    if alpha:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    h, w = img.shape[:2]

    # Calculate scale so image covers target size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #Crop to fit, ensuring constant size for all outputs
    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]

    image_cache[image_path] = cropped
    return cropped.copy()


video_cache = {}
def vid_load(video_path, progress, video_size = VIDEO_SIZE, alpha = False, make_alpha = False):
    if alpha:
        if progress == 0 and not make_alpha:
            return load(video_path.replace(".mp4", "_front.png").replace('/RAW/', '/RB/'), alpha=True)
        elif progress == 1 and not make_alpha:
            return load(video_path.replace(".mp4", "_back.png").replace('/RAW/', '/RB/'), alpha=True)
        elif make_alpha:
            folder_path = video_path.replace(".mp4","")
            os.makedirs(folder_path, exist_ok=True)
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            num_files = len(file_paths)
            if num_files == 0:
                rembg_video.rembg_video(video_path.replace('/RB/','/RAW/'), folder_path)
                file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                num_files = len(file_paths)


            def my_func(item):
                return (len(os.path.basename(item)) * 1_000) + int(os.path.basename(item).replace(".png",""))
            
            file_paths = sorted(file_paths, key=my_func)
                
            return load(file_paths[int(progress * (num_files - 1))], alpha = True)           

        else:
            raise Exception("Error: Transition Registry incorrect clearly -> correct the RB settings")

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
    return cropped.copy()

def smartload(image_path, block_progress, alpha = False, make_alpha = False):
    if image_path.endswith(".png"):
        return load(image_path, alpha = alpha)
    else:
        return vid_load(image_path, block_progress, alpha = alpha, make_alpha = make_alpha)

def nearest_dist(p,beats):
    return min([abs(x-p) for x in beats])

def current_beat_ind(p, beats):
    for i,x in enumerate(beats):
        if p <= x:
            return i
    return len(beats)

def get_font_path(font):
    return f"assets/fonts/{font.replace(' ', '')}.ttf"

def safe_input(text):
    if ENABLE_TRANSITION_PROMPTS:
        return input(text)
    print("\n\n")
    return None


def deterministic_8digit(s: str) -> str:
    # Hash the string with SHA256
    h = hashlib.sha256(s.encode()).hexdigest()

    num = int(h, 16)
    return int(str(num % 10**8).zfill(10))


#Transition modifiers 
# These are modifiers that can be applied on transition, all of these are wrappers
def pulse_on_beat(PULSE_DURATION = 0.05, INTENSITY = .25, BRIGHTNESS_BOOST = .25):
    def wrapper(my_func):
        def new_func(f1,f2,p, beats, total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)

            time_to_next_beat = nearest_dist(p,beats) * total_time
            if time_to_next_beat <= min(PULSE_DURATION, total_time):
                pulse_progress = (min(PULSE_DURATION, total_time) - time_to_next_beat)/min(PULSE_DURATION, total_time)
                pulse_progress = pulse_progress * .5
                return pulse_transform(frame, pulse_progress, INTENSITY, BRIGHTNESS_BOOST)

            return frame
        return new_func
    return wrapper

def camera_shake_on_beat(PULSE_DURATION = 0.05, INTENSITY = 50, FREQUENCY = 3):
    def wrapper(my_func):
        def new_func(f1,f2,p, beats, total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            
            time_to_next_beat = nearest_dist(p,beats) * total_time
            if time_to_next_beat <= min(PULSE_DURATION, total_time):
                pulse_progress = (min(PULSE_DURATION, total_time) - time_to_next_beat)/min(PULSE_DURATION, total_time)
                return camera_shake(frame, pulse_progress, intensity=50, frequency=3)

            return frame
        return new_func
    return wrapper

def vhs_filter(jitter_intensity=1, color_bleed=10, noise_strength=0.2):
    def wrapper(my_func):
        def new_func(f1,f2,p,beats,total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            return vhs_filter_transform(frame, p, jitter_intensity, color_bleed, noise_strength)
        return new_func
    return wrapper

def glitch_filter(max_shift=10, block_size=20):
    def wrapper(my_func):
        def new_func(f1,f2,p,beats,total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            return glitch_transform(frame, p, max_shift, block_size)
        return new_func
    return wrapper

def film_grain_filter(grain_strength=0.5, flicker_strength=0.05):
    def wrapper(my_func):
        def new_func(f1,f2,p,beats,total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            return film_grain(frame, p, grain_strength, flicker_strength)
        return new_func
    return wrapper

def heat_filter(amplitude=10, frequency=15, speed = 3):
    def wrapper(my_func):
        def new_func(f1,f2,p,beats,total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            return heat_distortion(frame, p, amplitude, frequency, speed)
        return new_func
    return wrapper

def chromatic_pulse_on_beat(PULSE_DURATION=0.1, INTENSITY=0.3):
    def wrapper(my_func):
        def new_func(f1,f2,p, beats, total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            
            time_to_next_beat = nearest_dist(p,beats) * total_time
            if time_to_next_beat <= min(PULSE_DURATION, total_time):
                pulse_progress = (min(PULSE_DURATION, total_time) - time_to_next_beat)/min(PULSE_DURATION, total_time)
                pulse_progress = pulse_progress * .5
                return chromatic_pulse(frame, pulse_progress, INTENSITY)

            return frame
        return new_func
    return wrapper

def motion_blur(motion_blur_total_frames = 5, motion_blur_time_diff = .05, start_prog = 0, end_prog = 1, start_prog_ind = None, end_prog_ind = None):
    """
    motion_blur_total_frame: Total frames to blur with
    motion_blur_time_diff = time_diff to blur with
    start_prog_ind = beat_index to start at
    end_prog_ind = beat_index to end at
    start_prog = start during/after this (blurring frames will also not include stuff outside this range)
    end_prog = end during/after this (blurring frames will also not include stuff outside this range)
    """
    def wrapper(my_func):
        def new_func(f1,f2,p, beats, total_time, *args):
            if start_prog_ind is not None:
                start_prog2 = beats[start_prog_ind]
            else:
                start_prog2 = start_prog
            
            if end_prog_ind is not None:
                end_prog2 = beats[end_prog_ind]
            else:
                end_prog2 = end_prog
            
            
            if p < start_prog2 or p > end_prog2:
                return my_func(f1,f2,p,beats,total_time, *args)

            my_frames = []
            motion_blur_p_diff = motion_blur_time_diff/total_time

            for x in range(-motion_blur_total_frames//2 + 1, motion_blur_total_frames//2 + 1):    
                current_p = p + (x * motion_blur_p_diff) #Get new p
                
                #Clip p
                current_p = max(start_prog2, current_p)
                current_p = min(current_p, end_prog2)
                
                frame = my_func(f1,f2,current_p,beats,total_time, *args)
                
                my_frames.append(frame.copy())
                
            return motion_blur_temporal(my_frames)
        return new_func
    return wrapper

def active_motion_blur(motion_blur_total_frames = 5, motion_blur_time_diff = .05):
    """
    motion_blur_total_frame: Total frames to blur with
    motion_blur_time_diff = time_diff to blur with

    transition function must reture -> frame, part_of_motion

    part_of_motion -> if True will be used as part of motion blur, if False will not
    """
    def wrapper(my_func):
        def new_func(f1,f2,p, beats, total_time, *args):
            frame, blur = my_func(f1,f2,p,beats,total_time, *args)

            if not blur:
                return frame

            my_frames = []
            motion_blur_p_diff = motion_blur_time_diff/total_time

            for x in [i for i in range(-motion_blur_total_frames//2 + 1, 0)][::-1]:
                current_p = p + (x * motion_blur_p_diff) #Get new p
                
                current_p = max(0, current_p)
                current_p = min(current_p, 1)
                
                frame, blur = my_func(f1,f2,current_p,beats,total_time, *args)
                if not blur:
                    break
                
                my_frames.append(frame.copy())
            
            my_frames = my_frames[::-1]

            for x in range(0, motion_blur_total_frames//2 + 1):
                current_p = p + (x * motion_blur_p_diff)
                
                current_p = max(0, current_p)
                current_p = min(current_p, 1)
                
                frame, blur = my_func(f1,f2,current_p,beats,total_time, *args)
                if not blur:
                    break
                
                my_frames.append(frame.copy())

                
            return motion_blur_temporal(my_frames)
        return new_func
    return wrapper

def breathe_on_beat(TIME_EACH = .11, MAX_ZOOM = 2, direction = "horizontal", BLUR = .1):
    horizontal = (direction == "horizontal")
    def wrapper(my_func):
        def new_func(f1,f2,p,beats,total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)

            stripes_seen = current_beat_ind(p, beats)            
            if stripes_seen == 0:
                time_for_this_beat = beats[0]
            elif stripes_seen == len(beats):
                return frame
            else:
                time_for_this_beat = beats[stripes_seen] - beats[stripes_seen - 1]
            time_for_this_beat = min(TIME_EACH/total_time, time_for_this_beat)  
            
            if 0 <= ((beats[stripes_seen] if stripes_seen < len(beats) else 1) - p) <= time_for_this_beat:
                current_prog = ((beats[stripes_seen] if stripes_seen < len(beats) else 1) - p)/time_for_this_beat
                current_prog = 1 - current_prog
                blur_prog = progress_func.parabolic_tepee(current_prog)

                zoom = (progress_func.linear(current_prog) * (MAX_ZOOM - 1)) + 1
                frame = transformers.zoom_transform_movement(frame, zoom=zoom, direction=[1,0] if horizontal else [0,1])

                if BLUR!= 0:
                    frame = add_extra_motion_blur(frame, BLUR * blur_prog, angle_deg=90 if horizontal else 0)

            return frame
        return new_func
    return wrapper


#All 3 good options to try out with various transitions
#def contour_flash(TIME_PERIOD = .06, COLOR = None, SOLID = (0,0,0), SPEED = 0, num_flashes = 1000000):
#def contour_flash(TIME_PERIOD = .06, COLOR = None, SOLID = (0,0,0), SPEED = 0, num_flashes = 2):
def contour_flash(TIME_PERIOD = 10000, COLOR = None, SOLID = (0,0,0), SPEED = 0, num_flashes = 2):
    """
    TIME_PERIOD: TP between flashes
    COLOR = contour color | None for rainbow color
    SOLID = Color of background | None for no background
    SPEED = Glaze speed
    """
    def wrapper(my_func):
        def new_func(f1,f2,p,beats,total_time, *args):
            frame = my_func(f1,f2,p,beats,total_time, *args)
            current_iter = ((p - beats[0])*total_time)//TIME_PERIOD
            if current_iter%2 == 1 or not (-num_flashes <= current_iter <= num_flashes):
                return frame
            background = frame if SOLID == None else solid_color_frame(frame, SOLID)
            _,contour = transformers.create_animated_glow_frame_contour_neon(frame,1, phi=p*SPEED, blur_kernel_size=3)
            return add_overlay(background, contour, 0, 0)
        return new_func
    return wrapper


#All this stuff based on linear cut also works with videos
#Fix linear and stuff to correctly split, based on beats
#  Transitions start from here  #

def linear_cut(f1,f2,p, beats, total_time, block_progress, alpha = False):
    if p < beats[0]:
        if f1.endswith(".png"):
            return load(f1, alpha=alpha)
        else:
            if alpha:
                block_progress = 1
            return vid_load(f1, block_progress[0], alpha=alpha)
    else:
        if f2.endswith(".png"):
            return load(f2, alpha=alpha)
        else:
            if alpha:
                block_progress = 0
            return vid_load(f2, block_progress[1], alpha=alpha)

def pulse_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    if abs(p - beats[0]) < .1:
        pulse_progress = ((.1 - abs(p - beats[0])) / .1)*.5
        return pulse_transform(frame, pulse_progress, intensity=0.4, brightness_boost=0.2)
    return frame

def large_pulse_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    if abs(p - beats[0]) < .1:
        pulse_progress = ((.1 - abs(p - beats[0])) / .1)*.5
        return pulse_transform(frame, pulse_progress, intensity=0.9, brightness_boost=0.3)
    return frame

def fade_to_black_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    if abs(p - beats[0]) < .1:
        pulse_progress = ((.1 - abs(p - beats[0])) / .1)*.5
        return fade_to_black(frame, pulse_progress)
    return frame

def fade_to_white_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    if abs(p - beats[0]) < .1:
        pulse_progress = ((.1 - abs(p - beats[0])) / .1)*.5
        return fade_to_white(frame, pulse_progress)
    return frame

def camera_shake_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    if abs(p - beats[0]) < .1:
        pulse_progress = (abs(p - beats[0]) / .1)*.5
        return camera_shake(frame, pulse_progress)
    return frame

#legacy, duck rotate is better
def rotate_pulse_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    rot_prog = 1 - (2*p)
    return zoom_rotate_transform(frame, rot_prog, theta=15)


def zoom_through_cut(f1,f2,p, beats, total_time, block_progress):
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    return zoom_transform(frame, (1 - (abs(0.5 - p) * 2))**6, max_zoom=10)

#legacy text-test + stripe test
def bouncy_stripes_cut(f1,f2,p, beats, total_time, block_progress):
    global TRANSITION_SETTINGS
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    name = TRANSITION_SETTINGS["NAME"]
    cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (140,140,140), 2, cv2.LINE_AA)

    return bouncy_stripes(frame, abs(0.5 - p) * 2, num_stripes=4, direction='vertical', glow_intensity=0.4, reveal_end=0.9, bounce_intensity=0.3)


def title_card(f1,f2,p,beats, total_time, block_progress):
    global TRANSITION_SETTINGS
    #frame = linear_cut(f1,f2,(p/beats[0])*.5 if p < beats[0] else (((p-beats[0])/(1-beats[0])) *.5) +.5001, [.5], total_time) #Rudimentary, maybe change later
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    name = str(TRANSITION_SETTINGS["NAME"])
    
    if p <= beats[0]:
        return frame

    i_fact = 0
    for i in range(0,4):
        if p < beats[i+1]:
            i_fact = int(i)
            break

    name = name[:math.ceil((i_fact/3) * len(name))]
    cv2.putText(frame, name, (200,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (140,140,140), 7, cv2.LINE_AA)

    return frame


# Helper functions
def _generate_synthetic_beats(beats, backward = False):
    if backward:
        beats = [0] + beats
        beats0 = []

        for i in range(len(beats) - 1):
            beats0.append((beats[i] + beats[i+1])/2)

        beats1 = []
        for i in range(len(beats0)):
            beats1.append(beats0[i])
            beats1.append(beats[i + 1])
            
        return beats1        

    else:
        beats.append(1)
        beats0 = []

        for i in range(len(beats) - 1):
            beats0.append((beats[i] + beats[i+1])/2)

        beats1 = []
        for i in range(len(beats0)):
            beats1.append(beats[i])
            beats1.append(beats0[i])

        return beats1        

def _get_best_slide_dv(fn, allow_diagonals = False):
    """Provides direction where movement should start from
    (x,y,z) -> all between +-2
    X -> +2 -> right side
    Y -> +2 -> downward
    Z -> +2 -> zooming in (smaller to bigger size)

    """
    if fn.endswith(".mp4"):
        return [-2*x for x in detect_average_motion(fn)]
    

    dvs_allowed = {'x':[], 'y':[],'z':[0]}
    sides = detect_empty_sides(smartload(fn.replace('/RAW/','/RB/'), 0, alpha=True))
    
    if sides["top"]:
        dvs_allowed['y'].append(2)
        #dvs_allowed['y'].append(random.randint(0,10)/100)
    if sides["bottom"]:
        dvs_allowed['y'].append(-2)
        #dvs_allowed['y'].append(random.randint(-10,0)/100)
    if sides["left"]:
        dvs_allowed['x'].append(2)
        #dvs_allowed['x'].append(random.randint(0,10)/100)
    if sides["right"]:
        dvs_allowed['x'].append(-2)
        #dvs_allowed['x'].append(random.randint(-10,0)/100)
    if sides["top"] and sides["bottom"] and sides["bottom"] and sides["bottom"]:
        dvs_allowed['z'].append(2)

    random.seed(fn)
    if len(dvs_allowed['x']) != 0 and len(dvs_allowed['y']) != 0:
        if random.randint(0,1) == 0:
            dvs_allowed['x'].append(0)
        else:
            dvs_allowed['y'].append(0)

    if len(dvs_allowed['x']) == 0:
        dvs_allowed['x'].append(0)
        if len(dvs_allowed['y']) == 0:
            dvs_allowed['y'].append(2)
    elif len(dvs_allowed['y']) == 0:
        dvs_allowed['y'].append(0)
    
    dvs = [random.choice(dvs_allowed['x']), random.choice(dvs_allowed['y']), random.choice(dvs_allowed['z'])]

    #Remove to allow diagonals
    dx,dy,dz = dvs
    if abs(dx) == 2 and abs(dy) == 2 and not allow_diagonals:
        if random.randint(0,1) == 0:
            dx = 0
        else:
            dy = 0
    
    return [dx,dy,dz]

def _cloud_bars_extra_slide_adv_raw(f1,f2,p,beats,total_time, block_progress, horizontal=False, NUM_BARS=2, padding = 0, alternate = False):
    #'NUM_BARS' Beats, perhaps as low as 120 BPM for this, but higher than that if possible
    bp1, bp2 = block_progress

    NUM_BARS *= 2
    beats = _generate_synthetic_beats(beats)


    if p > beats[-1]:
        frame = smartload(f2, bp2)
        overlay = alpha_add(smartload(f2, bp2), 1)
        return cloud_pulse_transform(frame, (p - beats[-1])/(1 - beats[-1]), overlay, MAX_SCALE=1.9, MAX_ALPHA=1)

    max_jitter = 0.05
    max_time = 0.15 #For the slide
    BLUR = 0.1

    if horizontal:
        direction = "horizontal"
        JITTER_Y = 1
        JITTER_X = 0

        DIRECTION_X = -1
        DIRECTION_Y = 0
    else:
        direction = "vertical"
        JITTER_X = 1
        JITTER_Y = 0

        DIRECTION_X = 0
        DIRECTION_Y = 1

    JITTER_X, JITTER_Y = JITTER_Y, JITTER_X

    overlay = smartload(f2.replace('/RAW/', '/RB/'), bp2, alpha=True)
    frame = smartload(f1, bp1)
    

    stripes_seen = current_beat_ind(p, beats)
    
    if alternate:
        DIRECTION_X, DIRECTION_Y = ((-1)**(stripes_seen)) * DIRECTION_X, ((-1)**(stripes_seen)) * DIRECTION_Y

    overlay, extra = bar_overlay_transformer_adv(overlay, padding=padding, inter_bar_padding=0, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction)
    if extra is not None: 
        if stripes_seen == 0:
            time_taken = beats[0] * total_time
            start_prog = 0
        else:
            time_taken = (beats[stripes_seen] - beats[stripes_seen - 1]) * total_time
            start_prog = beats[stripes_seen - 1]

        tru_time_for_transition = min(time_taken, max_time)
        tru_total_progress_units = tru_time_for_transition/total_time

        current_prog = (p - start_prog)/tru_total_progress_units

        current_prog = min(max(0, current_prog),1)

        bright = (1-current_prog)**.5

        current_prog = current_prog ** .2
        current_prog = 1 - current_prog

        jitter = max_jitter * current_prog
        slide = min(max(current_prog, 0), 1) * 2


        extra = add_brightness(extra, bright)
        overlay = add_overlay(overlay, extra, (jitter * 2 * JITTER_X)  + (slide * DIRECTION_X), (jitter * 2 * JITTER_Y) + (slide * DIRECTION_Y))

        if horizontal:
            overlay = add_extra_motion_blur(overlay, BLUR * slide * .5, angle_deg=0)
        else:
            overlay = add_extra_motion_blur(overlay, BLUR * slide * .5, angle_deg=90)

    return add_overlay(frame, overlay, 0, 0)


def cloud_bars(*args, **kwargs):
    return _cloud_bars_extra_slide_adv_raw(*args, **kwargs, horizontal=True, NUM_BARS=1, alternate=True)

def _cloud_bars_spawn_in(f1,f2,p,beats,total_time, block_progress, horizontal=False, padding = 0):
    #'NUM_BARS' Beats, perhaps as low as 120 BPM for this, but higher than that if possible
    NUM_BARS = len(beats)

    NUM_BARS *= 2
    beats = _generate_synthetic_beats(beats)

    
    bp1, bp2 = block_progress

    if p > beats[-1]:
        frame = smartload(f2, bp2)
        overlay = alpha_add(smartload(f2, bp2), 1)
        return cloud_pulse_transform(frame, (p - beats[-1])/(1 - beats[-1]), overlay, MAX_SCALE=1.9, MAX_ALPHA=1)

    BLUR = 0.09
    TIME_EACH = 0.1
    MAX_ZOOM = 2
    MAX_BRIGHT = .8

    if horizontal:
        direction = "horizontal"
    else:
        direction = "vertical"


    overlay = smartload(f2.replace('/RAW/', '/RB/'), bp2, alpha=True)
    frame = smartload(f1, bp1)
    

    stripes_seen = current_beat_ind(p, beats)
    
    overlay, extra = bar_overlay_transformer_adv(overlay, padding=padding, inter_bar_padding=0, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction)
    if extra is not None:
        if stripes_seen == 0:
            time_for_this_beat = beats[0]
        elif stripes_seen == len(beats):
            time_for_this_beat = 1 - beats[-1]
        else:
            time_for_this_beat = beats[stripes_seen] - beats[stripes_seen - 1]
        time_for_this_beat = min(TIME_EACH/total_time, time_for_this_beat)  
        
        if 0 <= (beats[stripes_seen] - p) <= time_for_this_beat:
            current_prog = (beats[stripes_seen] - p)/time_for_this_beat
            current_prog = 1 - current_prog
            blur_prog = progress_func.parabolic_tepee(current_prog)
            bright_prog = max(0,blur_prog)**.5

            zoom = (progress_func.linear(current_prog) * (MAX_ZOOM - 1)) + 1
            extra = transformers.zoom_object_wise_adv(extra, 1 if direction == "horizontal" else zoom, zoom if direction == "horizontal" else 1) 
            
            extra = add_brightness(extra, MAX_BRIGHT * bright_prog)
            extra = add_extra_motion_blur(extra, BLUR * blur_prog, angle_deg=90 if horizontal else 0)

            return add_overlay(add_overlay(frame, overlay, 0, 0), extra, 0, 0)
        
    return add_overlay(frame, overlay, 0, 0)


def cloud_bars_spawn_in(*args, **kwargs):
    #Adaptive function (2+ beats)
    return _cloud_bars_spawn_in(*args, **kwargs, horizontal=True)

def cloud_bars_spawn_in_vert(*args, **kwargs):
    #Adaptive function (2+ beats)
    return _cloud_bars_spawn_in(*args, **kwargs, horizontal=False)


def _cloud_bars_slide_in(f1,f2,p,beats,total_time, block_progress, horizontal=True, padding = 0):
    #'NUM_BARS' Beats, perhaps as low as 120 BPM for this, but higher than that if possible
    beats = _generate_synthetic_beats(beats)
    NUM_BARS = len(beats)
    
    bp1, bp2 = block_progress

    if p > beats[-1]:
        frame = smartload(f2, bp2)
        overlay = alpha_add(smartload(f2, bp2), 1)
        return cloud_pulse_transform(frame, (p - beats[-1])/(1 - beats[-1]), overlay, MAX_SCALE=1.9, MAX_ALPHA=1)

    BLUR = 0.09
    TIME_EACH = 0.1

    if horizontal:
        direction = "horizontal"
    else:
        direction = "vertical"


    overlay = smartload(f2.replace('/RAW/', '/RB/'), bp2, alpha=True)
    frame = smartload(f1, bp1)
    

    stripes_seen = current_beat_ind(p, beats)
    
    overlay, extra = bar_overlay_transformer_adv(overlay, padding=padding, inter_bar_padding=0, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction)
    if extra is not None:
        if stripes_seen == 0:
            time_for_this_beat = beats[0]
        elif stripes_seen == len(beats):
            time_for_this_beat = 1 - beats[-1]
        else:
            time_for_this_beat = beats[stripes_seen] - beats[stripes_seen - 1]
        time_for_this_beat = min(TIME_EACH/total_time, time_for_this_beat)  
        
        if 0 <= (beats[stripes_seen] - p) <= time_for_this_beat:
            current_prog = (beats[stripes_seen] - p)/time_for_this_beat
            current_prog = 1 - current_prog

            v = (1 - progress_func.linear(current_prog)) * 4 * ((stripes_seen%2) - .5)
            v = [v, 0] if direction == "horizontal" else [0,v]

            return add_overlay(add_overlay(frame, overlay, 0, 0), extra, v[0], v[1])
        
    return add_overlay(frame, overlay, 0, 0)

def cloud_bars_slide_in(*args, **kwargs):
    #Adaptive function (2+ beats)
    return _cloud_bars_slide_in(*args, **kwargs)

def cloud_pulse_slide(f1,f2,p,beats, total_time, block_progress):
    if len(beats) == 1:
        beats = [beats[0]/2, beats[0]]
    #Beats -> 2 (maybe 3 beats if BPM is too high)
    PAUSE = .95 #Possible cool effects if > 1
    PULSE_DURATION = 0.5 #Acutal duration = PULSE_DURATION * 2s
    BLUR = .05
    BLUR_DURATION = .05 #In seconds
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    bp1, bp2 = block_progress



    if p < beats[0]:
        frame = smartload(f1, bp1)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), bp2, alpha = True)
        slide_progress = (p/beats[0])
        slide_progress = progress_func.explosion(slide_progress) * PAUSE

        return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    elif beats[1] >= p >= beats[0]:
        frame = smartload(f1, bp1)
        
        slide_progress = ((p - beats[0])/(beats[1] - beats[0]))
        slide_progress = progress_func.explosion(slide_progress)
        slide_progress = PAUSE + (1-PAUSE) * slide_progress

        overlay = smartload(f2.replace('/RAW/', '/RB/'), bp2, alpha = True)

        frame = add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    else:
        frame = smartload(f2, bp2)


    pulse_start = beats[1]
    pulse_end = beats[1] + (PULSE_DURATION/total_time)
    pulse_start = max(0, pulse_start)
    pulse_end = min(1, pulse_end)

    blur_start = beats[1]
    blur_end = beats[1] + (BLUR_DURATION/total_time)
    blur_start = max(0, blur_start)
    blur_end = min(1, blur_end)



    if pulse_start <= p <= pulse_end:
        pulse_progress = (p - pulse_start) / (pulse_end - pulse_start)
        
        blur_progress = (p - blur_start) / (blur_end - blur_start)
        

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        #frame = shake_transform(frame, pulse_progress, intensity=.12, randomness=.05, angular_intensity=10) #Add back for extra blur
        frame = cloud_pulse_transform(frame, min(pulse_progress + .25, 1), overlay, MAX_SCALE=2, MAX_ALPHA=1, MIN_ALPHA=0)

    if blur_start <= p <= blur_end:
        frame = add_extra_motion_blur(frame, length_weighted=BLUR * 2 * progress_func.square(1 - abs(blur_progress - 0.5)), angle_deg=vector_to_angle_deg(DIRECTION_VECTOR))

    return frame

TEXT_CACHE = {}
SUBTITLE_TEXT_CACHE = {}
#Legacy, better to use the hook creator
#Known problem -> If you have a shorter character than a taller one, it 'glitches' kind of like sB -> will glitch sort of | This is fixed when using the hook creator
def _x_part_text_reveal_raw(f1,f2,p,beats,total_time,block_progress,
    BEATS = 3, #Give this many beats
    FONT_SIZE = .7,
    FONT = 'Anton',
    COLOR = "#FFFFFF",
    BORDER = "#000000",
    BORDER_SIZE = 0.022,
    SUBCAPTION = True, #Subtitle given the last beat -> its size = FONT_SIZE/2.2
    HIDE_BEHIND_OVERLAY = False, #Requires alpha = True; Hides main text (not subcaption, behind the main subject)
    GLASS_MODE = False, #Overrides most settings
    SHADOW = True):


    SUBTITLE_DAMPING = 3.2 #How much smaller the subtitle should be compared to actual one

    if p <= beats[0]:
        return smartload(f1, 1 if HIDE_BEHIND_OVERLAY else block_progress[0])

    global TEXT_CACHE
    if f2 in TEXT_CACHE:
        TEXT = TEXT_CACHE[f2]
    else:
        TEXT_CACHE[f2] = TEXT = input(f"\n\n\nEnter text wanted for {f2}: ")

    if SUBCAPTION:
        BEATS -= 1

    GLOW = None
    if GLASS_MODE:
        COLOR = "#000000"
        BORDER = None
        BORDER_SIZE = 0
        GLOW = None
        SHADOW = False
    
    frame = smartload(f2, 0 if HIDE_BEHIND_OVERLAY else block_progress[1])
    
    text_shown = TEXT[:math.ceil(min(1, current_beat_ind(p, beats)/BEATS) * len(TEXT))]
    
    text_overlay = create_text_overlay(text_shown,  get_font_path(FONT), reference_frame=frame, color=COLOR, border=BORDER, glow = GLOW, border_size=BORDER_SIZE, shadow=SHADOW, complete_text=TEXT) 
    final_overlay = create_text_overlay(TEXT,  get_font_path(FONT), reference_frame=frame, color=COLOR, border=BORDER, glow = GLOW,  border_size=BORDER_SIZE, shadow=SHADOW, complete_text=TEXT)
    
    
    final_overlay = resize_frame(final_overlay, FONT_SIZE, FONT_SIZE)
    resized_overlay = absolute_img_resize(text_overlay, final_overlay)

    new_overlay = add_overlay_inside_frame(alpha_add(final_overlay, 0), resized_overlay, x = -1, y = 0)

    if not GLASS_MODE:
        frame = add_overlay_inside_frame(frame, new_overlay, 0, 0.5 if SUBCAPTION else 0)
    else:
        glass_temp = add_overlay_inside_frame(alpha_add(frame, 0), new_overlay, 0, 0.5 if SUBCAPTION else 0)
        frame = glass_text_effect(frame, glass_temp)

    if HIDE_BEHIND_OVERLAY and not GLASS_MODE: #Invalid on glass mode
        overlay = smartload(f2.replace('/RAW/', '/RB/'), 1, alpha=True)
        frame = add_overlay(frame, overlay, 0, 0)

    if SUBCAPTION and p > beats[-1]:
        global SUBTITLE_TEXT_CACHE
        if f2 in SUBTITLE_TEXT_CACHE:
            SUBTITLE_TEXT = SUBTITLE_TEXT_CACHE[f2]
        else:
            SUBTITLE_TEXT_CACHE[f2] = SUBTITLE_TEXT = input(f"\n\n\nEnter in subtitle text for {f2}: ")
        if not GLASS_MODE:
            frame = add_overlay_inside_frame(frame, resize_frame(create_text_overlay(SUBTITLE_TEXT, font_path=get_font_path(FONT), reference_frame=frame, color=COLOR, border=BORDER, glow = GLOW,  border_size=BORDER_SIZE, shadow=SHADOW, complete_text=TEXT) ,FONT_SIZE/SUBTITLE_DAMPING, FONT_SIZE/SUBTITLE_DAMPING), 0, -.5)
        else:
            glass_temp = add_overlay_inside_frame(alpha_add(frame, 0), resize_frame(create_text_overlay(SUBTITLE_TEXT, font_path=get_font_path(FONT), reference_frame=frame, color=COLOR, border=BORDER, glow = GLOW,  border_size=BORDER_SIZE, shadow=SHADOW, complete_text=TEXT) ,FONT_SIZE/SUBTITLE_DAMPING, FONT_SIZE/SUBTITLE_DAMPING), 0, -.5)
            frame = glass_text_effect(frame, glass_temp)

    return frame

def four_part_text_reveal_anton_shadow_subtitle(*args, **kwargs):
    return _x_part_text_reveal_raw(*args, **kwargs, BEATS=4, FONT_SIZE=.7, FONT='Anton', SUBCAPTION=True, SHADOW=True, HIDE_BEHIND_OVERLAY=False)

def three_part_text_reveal_anton_shadow(*args, **kwargs):
    return _x_part_text_reveal_raw(*args, **kwargs, BEATS=3, FONT_SIZE=.7, FONT='Anton', SUBCAPTION=False, SHADOW=True, HIDE_BEHIND_OVERLAY=False)

def three_part_text_reveal_rampart_shadow(*args, **kwargs):
    return _x_part_text_reveal_raw(*args, **kwargs, BEATS=3, FONT_SIZE=.7, FONT='RampartOne', SUBCAPTION=True, SHADOW=True, HIDE_BEHIND_OVERLAY=False)

def four_part_text_reveal_anton_rb(*args, **kwargs):
    #Needs RB
    return _x_part_text_reveal_raw(*args, **kwargs, BEATS=4, FONT_SIZE=.7, FONT='Anton', SUBCAPTION=True, SHADOW=True, HIDE_BEHIND_OVERLAY=True)


def slide_cut(f1,f2,p, beats, total_time, block_progress, direction_vector=[2,2], MAX_BRIGHT = 0):
    MAX_BLUR = 0.15
    frame = smartload(f1, 1) if p < beats[0] else smartload(f2, 0)
    #Can take any number of beats, ideally 1 tho
    if p < beats[-1]:
        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        progress = min(1, p/beats[0])
        bright = progress_func.parabolic_tepee(progress) * MAX_BRIGHT
        progress = progress_func.quick(progress)
        
        overlay = add_brightness(overlay, bright)
        overlay = add_extra_motion_blur(overlay, length_weighted=bright * MAX_BLUR, angle_deg=vector_to_angle_deg(direction_vector))
        return add_overlay(frame, overlay, direction_vector[0] * (1 - progress), direction_vector[1] * (1 - progress))
    else:
        return frame
        #How to get a random direction vector, if needed:
        direction_vector = [x*y for x,y in zip(random.sample([2,0,2], 2), random.sample([-1,1,-1,1], 2))] #Random direction vector

def slide_cut_top(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[0,2])

def slide_cut_bottom(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[0,-2])

def slide_cut_left(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[-2,0])

def slide_cut_right(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[2,0])

#Better than above 4 since adaptive
def slide_cut_smart(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return slide_cut(*args, **kwargs, direction_vector=DIRECTION_VECTOR)


def slide_cut_top_bright(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[0,2], MAX_BRIGHT=.75)

def slide_cut_bottom_bright(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[0,-2], MAX_BRIGHT=.75)

def slide_cut_left_bright(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[-2,0], MAX_BRIGHT=.75)

def slide_cut_right_bright(*args, **kwargs):
    return slide_cut(*args, **kwargs, direction_vector=[2,0], MAX_BRIGHT=.75)

def slide_cut_smart_bright(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return slide_cut(*args, **kwargs, direction_vector=DIRECTION_VECTOR, MAX_BRIGHT=.75)


def radio_blur_cut(f1, f2, p, beats, total_time, block_progress, frame = None):
    #Beats -> 1
    BEAT_LENGTH = .2 #In seconds
    MAX_STRENGTH = 1.2
    SPEED = 0.05
    MAX_BLUR = 1.4 #Blur increases proprtional to square
    MAX_MOTION_BLUR = 2.2
    MAX_BRIGHTNESS = 1
    LINE_MIN_THICKNESS = .02

    rand_seed = deterministic_8digit(''.join(sorted([f1,f2])))
    random.seed(rand_seed)
    LINE_MIN_THICKNESS *= (1.25 - (.5 * random.random()))

    if frame is None:
        frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    beat_start = beats[0] - (BEAT_LENGTH/total_time) * .25 #because. ;)
    beats_end = beats[0] + (BEAT_LENGTH/total_time) * .75

    if beat_start <= p <= beats_end:
        line_pos = [round(random.random(),3) * (1-(3 * LINE_MIN_THICKNESS)) + LINE_MIN_THICKNESS for _ in range(3)]        
        
        beats_progress = (p - beat_start) / (beats_end - beat_start)
        
        beats_progress = progress_func.tepee(beats_progress)
        
        #strength_progress
        if 0 < beats_progress <= 0.12:
            strength_progress = (beats_progress/0.12) * 4.5
        elif beats_progress <= 0.22:
            strength_progress = 4.5
        elif beats_progress <= 0.31:
            strength_progress = 4.5 - ((beats_progress - 0.22)/(0.31 - 0.22))*4.5
        elif beats_progress <= 0.45:
            strength_progress = 0
        elif beats_progress <= 0.568:
            strength_progress = (beats_progress - 0.45)/(0.568 - 0.45) * 4.5
        elif beats_progress <= 0.8:
            strength_progress = 4.5
        elif beats_progress <= 0.9:
            strength_progress = 4.5 - ((beats_progress - 0.8)/(0.9 - 0.8)) * 4.5
        else:
            strength_progress = 0

        #blur progress:
        if beats_progress < 0.14 or beats_progress > 0.65:
            blur_progress = 0
        elif beats_progress > 0.35:
            blur_progress = -23*beats_progress*beats_progress + 16.1*beats_progress - 0.6175
        else:
            blur_progress = -49*beats_progress*beats_progress + 34.3*beats_progress - 3.8025
        blur_progress = blur_progress/100

        #length
        if 0.25 < beats_progress < 0.55:
            length_progress = 0.02
        else:
            length_progress = 0

        #brightness
        brightness_progress = (2/(1+math.exp(15 * (beats_progress - 1))) ) - 1

        #line progress range
        line_prog = (beats_progress - 0.3)/(0.7 - 0.3)
        if 0 <= line_prog <= 1:
            if line_prog < .2:
                line = (line_pos[0], line_pos[0] + LINE_MIN_THICKNESS)
            elif line_prog < .45:
                switch_prog = (line_prog - .2)/(.45 - .2)
                switch_prog = progress_func.square(switch_prog)
                if switch_prog < .5:
                    line = (line_pos[0], line_pos[0] + (line_pos[1] - line_pos[0]) * switch_prog * 2)
                else:
                    line = (line_pos[1], line_pos[1] + (line_pos[0] - line_pos[1]) * (1 - switch_prog) * 2)
                line = sorted(list(line))
                line = (line[0], line[1] + LINE_MIN_THICKNESS)
            elif line_prog < .55:
                line = (line_pos[2], line_pos[2] + LINE_MIN_THICKNESS)
            elif line_prog < .8:
                switch_prog = (line_prog - .55)/(.8 - .55)
                switch_prog = progress_func.square(switch_prog)
                if switch_prog < .5:
                    line = (line_pos[1], line_pos[1] + (line_pos[2] - line_pos[1]) * switch_prog * 2)
                else:
                    line = (line_pos[2], line_pos[2] + (line_pos[1] - line_pos[2]) * (1 - switch_prog) * 2)
                line = sorted(list(line))
                line = (line[0], line[1] + LINE_MIN_THICKNESS)

            else:
                line = (line_pos[1], line_pos[1] + LINE_MIN_THICKNESS)
        
            frame = whiteout_region(frame, line[0], line[1])

        frame = add_extra_motion_blur(frame, length_weighted=length_progress * MAX_MOTION_BLUR, angle_deg=30)
        frame = add_brightness(frame, .15 * brightness_progress * MAX_BRIGHTNESS)
        if blur_progress != 0:
            frame = add_blur(frame, ksize=blur_progress * MAX_BLUR, method="median")
        frame = heat_streak_distortion(frame, direction="horizontal", strength=MAX_STRENGTH*strength_progress, progress=beats_progress*SPEED, streak_size_x=1, streak_size_y=3, HEAT_SEED=rand_seed)
        
        return frame
    else:
        return frame

#legacy, probably better to use an intro double zoom then link via a radio blur cut
def double_zoom_cut_radio_blur(f1, f2, p, beats, total_time, block_progress):
    ZOOMS = [1, 1.25, 1.5, 1]
    zoom = ZOOMS[current_beat_ind(p, beats)]
    
    frame = linear_cut(f1, f2, p, [beats[-1]], total_time, block_progress)
    frame = zoom_transform(frame, 1, zoom)

    if p > beats[-2]:
        frame = radio_blur_cut(f1, f2, p, [beats[-1]], total_time, block_progress, frame)

    return frame    


def black_slides(f1,f2,p,beats,total_time,block_progress):
    #2 Beats, requires rb
    MAX_SLIDE_SIZE = 0.3
    MAX_BRIGHT = 0.35

    if p > beats[1]:
        return smartload(f2, block_progress=0)

    slide_prog = p/beats[1]
    slide_prog = progress_func.parabolic_tepee(slide_prog)

    frame = smartload(f1, block_progress=1)
    overlay = smartload(f1.replace('/RAW/', '/RB/'), block_progress=1, alpha=True)

    overlay = zoom_transform(overlay,1, MAX_BRIGHT * slide_prog + 1)

    frame = whiteout_region(frame, 0, MAX_SLIDE_SIZE * slide_prog, (0,0,0), orientation="horizontal")
    frame = whiteout_region(frame, 1 - (MAX_SLIDE_SIZE * slide_prog), 1, (0,0,0), orientation="horizontal")

    return add_overlay(frame, overlay, 0, 0)

def slide_shake_brr(f1,f2,p,beats, total_time, block_progress):
    #Beats -> 2+
    SLIDE_DURATION = .34 #In seconds
    SHAKE_DURATION = 0.26
    STREAK = 3.5
    BLUR = 0.1
    PAUSE = .75 #%to pause at
    PAUSE_LEN = 0
    JITTER = .3
    SECONDARY_JITTER = JITTER * .75
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    DECAY = .8

    # ---

    if abs(dx) > abs(dy):
        direction = "horizontal"
    else:
        direction = "vertical"

    streak_size_x = 1
    streak_size_y = 15

    if direction == "vertical":
        streak_size_x, streak_size_y = streak_size_y, streak_size_x

    start_slide = 0 #beats[-1] - min(SLIDE_DURATION/total_time, beats[-1])

    #Extra shake settings
    center = 0.18 
    shake_len = SHAKE_DURATION/total_time
    shake_start = beats[-1] - center*shake_len
    shake_end = beats[-1] + (1-center)*shake_len
    sharpness = .8

    if p < beats[-1]:
        frame = smartload(f1, block_progress[0])
    else:
        frame = smartload(f2, block_progress[1]) 
        
    if start_slide <= p <= beats[-1]:
        slide_progress = progress_func.smooth_stop_in_between_extra_adv((p - start_slide)/(beats[-1] - start_slide), center_y_val=PAUSE, pause_dist=PAUSE_LEN)

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        frame = add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))

    if shake_start <= p <= shake_end:
        shake_progress = progress_func.asymmetric_parabolic_tepee((p - shake_start)/(shake_end - shake_start), center=center, l=sharpness)

    if shake_start <= p <= shake_end and JITTER > 0:
        pulse_progress = (p - shake_start)/(shake_end - shake_start) 
        
        dx, dy = 1 if direction == "horizontal" else SECONDARY_JITTER/JITTER, 1 if direction == "vertical" else SECONDARY_JITTER/JITTER       
        
        dx, dy = progress_func.damped_jitter(
            pulse_progress,
            jitter_x=dx, jitter_y=dy,
            cycles_x=5, cycles_y=4,
            seed = f1 + f2, 
            warp=1.4, amp_decay=1, freq_spread=.12
        )
        dx, dy = JITTER * dx, JITTER * dy
        
        frame =  transformers.raw_shake(frame, direction_vector=(dx,dy))


    if shake_start <= p <= shake_end and STREAK > 0:
        frame = transformers.heat_streak_distortion(frame, direction, strength=STREAK*shake_progress, streak_size_x=streak_size_x, streak_size_y=streak_size_y)

    if shake_start <= p <= shake_end and BLUR > 0:
        frame = transformers.add_extra_motion_blur(frame, length_weighted=BLUR * shake_progress, angle_deg=0 if direction == "horizontal" else 90)

    return frame



#@contour_flash()
def shake_brr(f1,f2,p,beats, total_time, block_progress):
    #Beats -> 1
    SHAKE_DURATION = 0.15
    STREAK = 3.5
    BLUR = 0.1
    JITTER = .3
    SECONDARY_JITTER = JITTER * .75
    direction = "horizontal"
    
    streak_size_x = 1
    streak_size_y = 15

    if direction == "vertical":
        streak_size_x, streak_size_y = streak_size_y, streak_size_x

    #Extra shake settings
    center = 0.18 
    shake_len = SHAKE_DURATION/total_time
    shake_start = beats[0] - center*shake_len
    shake_end = beats[0] + (1-center)*shake_len
    sharpness = .8

    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
    else:
        frame = smartload(f2, block_progress[1]) 
        
    if shake_start <= p <= shake_end:
        shake_progress = progress_func.asymmetric_parabolic_tepee((p - shake_start)/(shake_end - shake_start), center=center, l=sharpness)

    if shake_start <= p <= shake_end and JITTER > 0:
        pulse_progress = (p - shake_start)/(shake_end - shake_start) 
        
        dx, dy = 1 if direction == "horizontal" else SECONDARY_JITTER/JITTER, 1 if direction == "vertical" else SECONDARY_JITTER/JITTER       
        
        dx, dy = progress_func.damped_jitter(
            pulse_progress,
            jitter_x=dx, jitter_y=dy,
            cycles_x=5, cycles_y=4,
            seed = f1 + f2, 
            warp=1.4, amp_decay=1, freq_spread=.12
        )
        dx, dy = JITTER * dx, JITTER * dy
        
        frame =  transformers.raw_shake(frame, direction_vector=(dx,dy))


    if shake_start <= p <= shake_end and STREAK > 0:
        frame = transformers.heat_streak_distortion(frame, direction, strength=STREAK*shake_progress, streak_size_x=streak_size_x, streak_size_y=streak_size_y)

    if shake_start <= p <= shake_end and BLUR > 0:
        frame = transformers.add_extra_motion_blur(frame, length_weighted=BLUR * shake_progress, angle_deg=0 if direction == "horizontal" else 90)

    return frame

def shake_brr22(f1,f2,p,beats, total_time, block_progress):
    #Beats -> 1
    SHAKE_DURATION = 0.15
    STREAK = 3.5
    BLUR = 0.1
    JITTER = .3
    SECONDARY_JITTER = JITTER * .75
    direction = "horizontal"
    
    streak_size_x = 1
    streak_size_y = 15

    if direction == "vertical":
        streak_size_x, streak_size_y = streak_size_y, streak_size_x

    #Extra shake settings
    center = 0.18 
    shake_len = SHAKE_DURATION/total_time
    shake_start = beats[0] - center*shake_len
    shake_end = beats[0] + (1-center)*shake_len
    sharpness = .8

    if p < beats[0]:
        frame = f1
    else:
        frame = f2
        
    if shake_start <= p <= shake_end:
        shake_progress = progress_func.asymmetric_parabolic_tepee((p - shake_start)/(shake_end - shake_start), center=center, l=sharpness)

    if shake_start <= p <= shake_end and JITTER > 0:
        pulse_progress = (p - shake_start)/(shake_end - shake_start) 
        
        dx, dy = 1 if direction == "horizontal" else SECONDARY_JITTER/JITTER, 1 if direction == "vertical" else SECONDARY_JITTER/JITTER       
        
        dx, dy = progress_func.damped_jitter(
            pulse_progress,
            jitter_x=dx, jitter_y=dy,
            cycles_x=5, cycles_y=4,
            seed = 0, 
            warp=1.4, amp_decay=1, freq_spread=.12
        )
        dx, dy = JITTER * dx, JITTER * dy
        
        frame =  transformers.raw_shake(frame, direction_vector=(dx,dy))


    if shake_start <= p <= shake_end and STREAK > 0:
        frame = transformers.heat_streak_distortion(frame, direction, strength=STREAK*shake_progress, streak_size_x=streak_size_x, streak_size_y=streak_size_y)

    if shake_start <= p <= shake_end and BLUR > 0:
        frame = transformers.add_extra_motion_blur(frame, length_weighted=BLUR * shake_progress, angle_deg=0 if direction == "horizontal" else 90)

    return frame


def shake_brr_raw(f1,f2,p,beats, total_time, block_progress, SHAKE_DURATION = .26):
    #Beats -> 1
    STREAK = 3.5
    BLUR = 0.1
    JITTER = .3
    SECONDARY_JITTER = JITTER * .75
    direction = "horizontal"
    
    streak_size_x = 1
    streak_size_y = 15

    if direction == "vertical":
        streak_size_x, streak_size_y = streak_size_y, streak_size_x

    #Extra shake settings
    center = 0.18 
    shake_len = SHAKE_DURATION/total_time
    shake_start = beats[0] - center*shake_len
    shake_end = beats[0] + (1-center)*shake_len
    sharpness = .8

    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
    else:
        frame = smartload(f2, block_progress[1]) 
        
    if shake_start <= p <= shake_end:
        shake_progress = progress_func.asymmetric_parabolic_tepee((p - shake_start)/(shake_end - shake_start), center=center, l=sharpness)

    if shake_start <= p <= shake_end and JITTER > 0:
        pulse_progress = (p - shake_start)/(shake_end - shake_start) 
        
        dx, dy = 1 if direction == "horizontal" else SECONDARY_JITTER/JITTER, 1 if direction == "vertical" else SECONDARY_JITTER/JITTER       
        
        dx, dy = progress_func.damped_jitter(
            pulse_progress,
            jitter_x=dx, jitter_y=dy,
            cycles_x=5, cycles_y=4,
            seed = f1 + f2, 
            warp=1.4, amp_decay=1, freq_spread=.12
        )
        dx, dy = JITTER * dx, JITTER * dy
        
        frame =  transformers.raw_shake(frame, direction_vector=(dx,dy))


    if shake_start <= p <= shake_end and STREAK > 0:
        frame = transformers.heat_streak_distortion(frame, direction, strength=STREAK*shake_progress, streak_size_x=streak_size_x, streak_size_y=streak_size_y)

    if shake_start <= p <= shake_end and BLUR > 0:
        frame = transformers.add_extra_motion_blur(frame, length_weighted=BLUR * shake_progress, angle_deg=0 if direction == "horizontal" else 90)

    return frame


#Idk broken for some reason, doesn't look super good
@active_motion_blur(motion_blur_total_frames=8, motion_blur_time_diff=0.005)
def ripple_effect_cool_thing(f1, f2, p, beats, total_time, block_progress):
    #Adaptive, but ideally 3-7 beats,
    BEAT_TIME = .2 #For ripple
    BEAT_TIME2 = .1 #For 
    start = .2
    MAX_ZOOM = 1.3
    n_factor = .8

    do_motion_blur = False
    bm1 = max(0, beats[-1] - ((BEAT_TIME/total_time) * n_factor))
    sheen = None
    if p <= bm1:
        zoom = progress_func.asymmetric_parabolic_tepee(p * (1/bm1), center=beats[-2] * (1/bm1)) * (MAX_ZOOM - 1) + 1 #Zoom for all overlays
        sheen = progress_func.linear(p * (1/bm1))
    elif p < beats[-1]:
        zoom = 1

    if p > beats[-1]:
        frame = smartload(f2, block_progress=0)
    else:
        frame = smartload(f1, block_progress=1)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress=0, alpha=True)
                

        cb = current_beat_ind(p, beats)
        #bar_overlay, new_overlay = transformers.bar_overlay_transformer_adv(overlay, bars_visible=cb, num_bars=len(beats) - 1, padding=0, inter_bar_padding=0, direction="horizontal", tight_fit=True)
        bar_overlay, new_overlay = transformers.bar_overlay_transformer_adv2(overlay, bars_visible=cb, num_bars=len(beats) - 1, padding=0, inter_bar_padding=0, direction="horizontal")
        
        if sheen is not None:
            _, undoer,redoer = tight_fit_function_runner(overlay) #Get final bounding box
            bar_overlay = undoer(apply_sheen(redoer(bar_overlay), sheen))
            
        
        if zoom >= 1.5 or zoom <= .5:
            raise Exception("BROOOO!")
        frame = add_overlay(frame, zoom_transform(bar_overlay, 1, zoom), 0, 0)

        if new_overlay is not None:
            prev = beats[cb - 1] if cb != 0 else 0

            max_beat_length = total_time * BEAT_TIME2

            if beats[cb] - prev <= max_beat_length:
                bcb = beats[cb]
                prev2 = prev
            else:
                diff = ((beats[cb] - prev) - max_beat_length)/2
                bcb = beats[cb] - diff
                prev2 = prev + diff

            if p >= prev2:
                current_section_prog = (p - prev2)/(bcb - prev2)
                current_section_prog = min(1, current_section_prog)

                new_overlay = add_brightness(new_overlay, 1 - current_section_prog**2)
                new_overlay = add_extra_motion_blur(new_overlay, angle_deg=45)
                frame = add_overlay(frame, zoom_transform(new_overlay, 1, zoom), 1*(1 - current_section_prog), -1*(1 - current_section_prog))

                do_motion_blur = True



    if abs(p - beats[-1]) <= BEAT_TIME/total_time:
        prog = (p - (beats[-1] - (BEAT_TIME/total_time)))/(2 * (BEAT_TIME/total_time))
        frame = add_brightness(frame, progress_func.parabolic_tepee(prog) ** 2)
        return droplet_effect(frame, start + (prog * (1-start))), do_motion_blur
    
    return frame, do_motion_blur
    

def ripple_cut(f1, f2, p, beats, total_time, block_progress):
    #Adaptive, but ideally 3-7 beats, #Yeah I dont think this is correct past me was wrong, 1 beat should be enough because of the video snapping
    frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    BEAT_TIME = .2 #For ripple
    start = .05
    start2 = .1 #More than start, second ring

    if abs(p - beats[-1]) <= BEAT_TIME/total_time:
        prog = (p - (beats[-1] - (BEAT_TIME/total_time)))/(2 * (BEAT_TIME/total_time))
        frame = add_brightness(frame, progress_func.tepee(prog) ** 3)
        return droplet_effect(droplet_effect(frame, start + (prog * (1-start)), width=0.05), min(1, start2 + (prog * (1-start))), width=0.05)
    
    return frame
    
#Yooo these both actually look sick ngl. Its been hours, its 3AM and it was 1000% worth it.
def glitch_cut(f1,f2,p,beats,total_time, block_progress):
    BEAT_LENGTH = .25
    MAX_GLITCH = .2
        
    if p <= beats[0]:
        frame = smartload(f1, 1)
        overlay = smartload(f1.replace("/RAW/","/RB/"), 1, alpha=True)
    else:
        frame = smartload(f2, 0)
        overlay = smartload(f2.replace("/RAW/","/RB/"), 0, alpha=True)

    BEAT_LENGTH = BEAT_LENGTH/(2 * total_time)

    if abs(p - beats[-1]) <= BEAT_LENGTH:
        prog = 1 - (abs(p - beats[-1])/BEAT_LENGTH)

        overlay = glitch_transform(overlay, prog * MAX_GLITCH)
        #frame = glitch_transform(alpha_add(frame, 1), prog * MAX_GLITCH)
        frame = add_overlay(frame, overlay, 0, 0)

    return frame

def glitch_cut2(f1,f2,p,beats,total_time, block_progress):
    BEAT_LENGTH = .15
    MAX_GLITCH = .05
        
    if p <= beats[0]:
        frame = smartload(f1, 1)
        overlay = smartload(f1.replace("/RAW/","/RB/"), 1, alpha=True)
    else:
        frame = smartload(f2, 0)
        overlay = smartload(f2.replace("/RAW/","/RB/"), 0, alpha=True)

    BEAT_LENGTH = BEAT_LENGTH/(2 * total_time)

    if abs(p - beats[-1]) <= BEAT_LENGTH:
        prog = 1 - (abs(p - beats[-1])/BEAT_LENGTH)
        overlay = hologram_effect(overlay, prog * MAX_GLITCH)
        frame = add_overlay(frame, overlay, 0, 0)

    return frame

def my_neon_template(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 4 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    beats = _generate_synthetic_beats(beats, backward = True)
    beats = _generate_synthetic_beats(beats, backward = True)
    
    color = (255,255,255)
    if p >= beats[-1]:
        return smartload(f2, 0)
    
    frame = smartload(f1, 1)
    overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)

    if p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), 0, alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)


    if p < beats[0]:
        overlay2 = silhouette(overlay2, color)
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[3]:
        return add_overlay(frame, overlay, 0, 0)

def neon_contour_switch_up(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 1 beat

    Normal, Normal + Contour, Black + contour (wisping -> reversed + overlayed w/ new contour), Normal + contour, Normal
    """
    x1,x2 = .2, .4
    TIME_STAMPS = [x1, x2, 1 -x2, 1-x1]
    MAX_TIME = .5 #In seconds
    PHI = 2


    actual_time = min(MAX_TIME, total_time)/total_time #See actual given progress units
    TIME_STAMPS = [x*actual_time + ((1-actual_time)/2) for x in TIME_STAMPS]
   
    frame1 = smartload(f1, block_progress[0])
    frame2 = smartload(f2, block_progress[1])
    
    
    if p < TIME_STAMPS[0]:
        return frame1
    elif p < TIME_STAMPS[1]:
        frame, contour = transformers.create_animated_glow_frame_contour_neon(frame1, 1, phi=p*PHI)
        #return frame
        return add_overlay(solid_color_frame(frame1, (0,0,0)), contour, 0, 0)

    elif p < TIME_STAMPS[2]:
        inner_transition_progress = 1 - (p - TIME_STAMPS[1])/(TIME_STAMPS[2] - TIME_STAMPS[1])
        _, contour = transformers.create_animated_glow_frame_contour_neon(frame1, inner_transition_progress, phi=p*PHI)
        _, contour_2 = transformers.create_animated_glow_frame_contour_neon(frame2, 1 - inner_transition_progress, phi=p*PHI)
        return add_overlay(solid_color_frame(frame1, (0,0,0)), add_overlay(contour, contour_2, 0, 0), 0, 0)
    elif p < TIME_STAMPS[3]:
        frame, contour = transformers.create_animated_glow_frame_contour_neon(frame2, 1, phi=p*PHI)
        #return frame
        return add_overlay(solid_color_frame(frame1, (0,0,0)), contour, 0, 0)

    else:
        return frame2
    
    
def colour_cut(f1,f2,p,beats,total_time, block_progress):
    BEAT_LENGTH = .15
    TIME_PER_COLOUR = .05 #In seconds

    #Add more if wanted. Currently these all look great on most images
    COLOUR_PALLETES = [
        [(60, 15, 10), (69, 10, 59), (162, 31, 122), (0, 215, 255), (204, 247, 255)],
        [(10, 30, 10), (20, 100, 40), (50, 200, 100), (0, 255, 255), (200, 255, 200)],
        [(10, 10, 30), (60, 0, 120), (120, 0, 255), (180, 80, 255), (255, 240, 255)],
        [(0, 0, 0), (0, 80, 0), (0, 255, 0), (0, 255, 255), (255, 255, 255)],
        [(0, 0, 0), (0, 0, 120), (0, 80, 255), (0, 200, 255), (255, 255, 255)],
        [(200, 160, 255), (255, 130, 200), (180, 200, 255), (200, 255, 230), (255, 255, 255)],
        [(0, 0, 0), (40, 0, 80), (100, 0, 160), (0, 0, 255), (0, 120, 255)],
        [(0, 0, 0), (0, 180, 0), (180, 255, 255), (180, 0, 180), (255, 255, 255)],
        [(100, 40, 200), (180, 80, 255), (255, 150, 220), (200, 255, 230), (255, 255, 255)],
        [(0, 0, 0), (30, 30, 40), (0, 0, 180), (0, 128, 255), (0, 215, 255)],
    ]

    random.seed(f1 + f2)
    random.shuffle(COLOUR_PALLETES)
    COLOUR_PALLETES = COLOUR_PALLETES[:3]

    if p <= beats[0]:
        frame = smartload(f1, 1)
    else:
        frame = smartload(f2, 0)

    BEAT_LENGTH = BEAT_LENGTH/(2 * total_time)

    if abs(p - beats[-1]) <= BEAT_LENGTH:
        prog = ((p - (beats[-1] - BEAT_LENGTH))/(2 * BEAT_LENGTH))
        t = prog * total_time
        current_col = t//TIME_PER_COLOUR
        current_pallete = COLOUR_PALLETES[int(current_col)%len(COLOUR_PALLETES)]

        frame = color_tint_gradient_map(frame, colors=current_pallete)


    return frame



def zoom_pan_in(f1,f2,p,beats,total_time, block_progress):
    MAX_ZOOM = 2.15

    frame = linear_cut(f1,f2,p,beats, total_time, block_progress)
    if p < beats[-1]:
        zoom = (MAX_ZOOM-1)*progress_func.square(p/beats[-1]) + 1
        frame = zoom_transform(frame, 1, zoom)
        frame = radial_blur(frame, 50 * progress_func.square(progress_func.square(p/beats[-1])))
    
    return frame

@motion_blur(3, 0.01)
def zoom_pan_in_non_chalant(f1,f2,p,beats,total_time, block_progress):
    MAX_ZOOM = 2.15

    frame = linear_cut(f1,f2,p,beats, total_time, block_progress)
    if p < beats[-1]:
        zoom = (MAX_ZOOM-1)*progress_func.square(p/beats[-1]) + 1
        frame = zoom_transform(frame, 1, zoom)
    
    return frame


@motion_blur(3, 0.01)
def zoom_pan_in_bounce(f1,f2,p,beats,total_time, block_progress):
    MAX_ZOOM = 2.15

    frame = linear_cut(f1,f2,p,beats, total_time, block_progress)
    if p < beats[-1]:
        zoom = (MAX_ZOOM-1)*progress_func.square(p/beats[-1]) + 1
        frame = zoom_transform(frame, 1, zoom)
    else:
        zoom = (MAX_ZOOM-1)*progress_func.square(1 - ((p-beats[-1])/(1-beats[-1]))) + 1
        frame = zoom_transform(frame, 1, zoom)
    
    return frame



def white_cut(f1,f2,p,beats,total_time, block_progress):
    BEAT_LENGTH = .15
    MAX_WHITE = .45
    frame = linear_cut(f1,f2,p,beats,total_time, block_progress)
    
    BEAT_LENGTH = BEAT_LENGTH/(2 * total_time)

    if abs(p - beats[-1]) <= BEAT_LENGTH:
        prog = 1 - (abs(p - beats[-1])/BEAT_LENGTH)
        frame = fade_to_white(frame, progress=prog * MAX_WHITE)

    return frame


def lens_blur_cut(f1,f2,p,beats,total_time, block_progress):
    BEAT_LENGTH = .4
    frame = linear_cut(f1,f2,p,beats,total_time, block_progress)
    
    BEAT_LENGTH = min(total_time, BEAT_LENGTH)/(2 * total_time)

    if abs(p - beats[-1]) <= BEAT_LENGTH:
        prog = 1 - (abs(p - beats[-1])/BEAT_LENGTH)
        frame = apply_lens_blur(frame, strength=prog)

    return frame


def bright_cut(f1,f2,p,beats,total_time, block_progress):
    BEAT_LENGTH = .15
    MAX_WHITE = .3
    frame = linear_cut(f1,f2,p,beats,total_time, block_progress)
    
    BEAT_LENGTH = BEAT_LENGTH/(2 * total_time)

    if abs(p - beats[-1]) <= BEAT_LENGTH:
        prog = 1 - (abs(p - beats[-1])/BEAT_LENGTH)
        frame = add_brightness(frame, prog * MAX_WHITE)

    return frame

@breathe_on_beat(TIME_EACH = .11, MAX_ZOOM = 1.09, direction = "vertical", BLUR = .1)
def bright_cut_breathe(*args, **kwargs):
    return bright_cut(*args, **kwargs)

def trip_bars_padding(f1,f2,p,beats,total_time, block_progress):
    #4 beats extend_block_progress = True for more 'immersiveness' -> perhaps turn it auto on if possible for anything not require rb
    if p > beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])

    bars_visible = current_beat_ind(p, beats)
    bars_overlay, _ = bar_overlay_transformer_adv(smartload(f2, block_progress[1]), padding=0.0375, inter_bar_padding=0.0375, bars_visible=bars_visible, num_bars=3)

    return add_overlay(frame, bars_overlay,0,0)

#Helper function
def _onebeat_back_beatsynth(beats, back, front):
    l1 = beats[0]
    l2 = 1 - beats[0]

    new_beats = []

    dist1 = l1/(back+1)
    new_beats.extend([x*dist1 for x in range(1, back + 1)])

    new_beats.append(beats[0])

    dist2 = min(l2/(front+1), dist1)
    new_beats.extend([x*dist2 + l1 for x in range(1, front + 1)])

    return new_beats

def trip_bars_padding_synth(f1,f2,p,beats,total_time, block_progress):
    beats = _onebeat_back_beatsynth([beats[-1]], 3, 0) #Adds 3 beats behind tru beat
    return trip_bars_padding(f1,f2,p,beats,total_time, block_progress)

def adaptive_bars_lean(f1,f2,p,beats,total_time, block_progress):
    #x + 1 beats, x = number of bars extend_block_progress = True for more 'immersiveness' -> perhaps turn it auto on if possible for anything not require rb
    beats = _generate_synthetic_beats(beats)
    if p > beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])

    bars_visible = current_beat_ind(p, beats)
    bars_overlay, _ = bar_overlay_transformer_adv(smartload(f2, block_progress[1]), padding=12/500, inter_bar_padding=12/500, bars_visible=bars_visible, num_bars=len(beats) - 1)

    return add_overlay(frame, bars_overlay,0,0)



def adaptive_bars_lean2(f1,f2,p,beats,total_time, block_progress):
    #x + 1 beats, x = number of bars extend_block_progress = True for more 'immersiveness' -> perhaps turn it auto on if possible for anything not require rb
    if p > beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])

    bars_visible = current_beat_ind(p, beats)
    bars_overlay, _ = bar_overlay_transformer_adv_diag(smartload(f2, block_progress[1]), padding=12/500, inter_bar_padding=12/500, bars_visible=bars_visible, num_bars=len(beats) - 1)

    return add_overlay(frame, bars_overlay,0,0)

def nine_bars_padding(f1,f2,p,beats,total_time, block_progress):
    #10 beats extend_block_progress = True for more 'immersiveness' -> perhaps turn it auto on if possible for anything not require rb
    if p > beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])

    bars_visible = current_beat_ind(p, beats)
    bars_overlay, _ = bar_overlay_transformer_adv(smartload(f2, block_progress[1]), padding=0.035, inter_bar_padding=0.025, bars_visible=bars_visible, num_bars=9)

    return add_overlay(frame, bars_overlay,0,0)


def _bars_extra_slide_raw(f1,f2,p,beats,total_time, block_progress, horizontal=False, NUM_BARS=5, padding = 0, alternate = False):
    #'NUM_BARS' Beats, if padding !=0, NUM_BARS + 1, perhaps as low as 120 BPM for this, but higher than that if possible
    max_jitter = 0.0
    max_time = 0.3

    if padding != 0 and p > beats[-1]:
        return smartload(f2, block_progress[1])

    if horizontal:
        direction = "horizontal"
        JITTER_Y = 1
        JITTER_X = 0

        DIRECTION_X = -1
        DIRECTION_Y = 0
    else:
        direction = "vertical"
        JITTER_X = 1
        JITTER_Y = 0

        DIRECTION_X = 0
        DIRECTION_Y = 1

    overlay = smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])

    stripes_seen = current_beat_ind(p, beats)
    
    if alternate:
        DIRECTION_X, DIRECTION_Y = ((-1)**(stripes_seen)) * DIRECTION_X, ((-1)**(stripes_seen)) * DIRECTION_Y

    overlay, extra = bar_overlay_transformer_adv(overlay, inter_bar_padding=0, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction, padding=padding)
    if extra is not None:
        if stripes_seen == 0:
            time_taken = beats[0] * total_time
            start_prog = 0
        else:
            time_taken = (beats[stripes_seen] - beats[stripes_seen - 1]) * total_time
            start_prog = beats[stripes_seen - 1]

        tru_time_for_transition = min(time_taken, max_time)
        tru_total_progress_units = tru_time_for_transition/total_time

        current_prog = (p - start_prog)/tru_total_progress_units

        current_prog = min(max(0, current_prog),1)
        bright = (1-current_prog)**.5


        current_prog = current_prog ** .3

        current_prog = 1 - current_prog

        jitter = max_jitter * current_prog
        slide = min(max(current_prog, 0), 1) * 2


        extra = add_brightness(extra, bright * .4)
        overlay = add_overlay(overlay, extra, (jitter * 2 * JITTER_X)  + (slide * DIRECTION_X), (jitter * 2 * JITTER_Y) + (slide * DIRECTION_Y))

    return add_overlay(frame, overlay, 0, 0)

def _bars_extra_slide_adv_raw(f1,f2,p,beats,total_time, block_progress, horizontal=False, NUM_BARS=2, padding = 0, alternate = False):
    #'NUM_BARS + 1' Beats, perhaps as low as 120 BPM for this, but higher than that if possible
    
    if p > beats[-1]:
        return smartload(f2, 0)

    max_jitter = 0.05
    max_time = 0.3
    BLUR = 0.1

    if horizontal:
        direction = "horizontal"
        JITTER_Y = 1
        JITTER_X = 0

        DIRECTION_X = -1
        DIRECTION_Y = 0
    else:
        direction = "vertical"
        JITTER_X = 1
        JITTER_Y = 0

        DIRECTION_X = 0
        DIRECTION_Y = 1

    JITTER_X, JITTER_Y = JITTER_Y, JITTER_X

    overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha=True)
    frame = smartload(f1, 1)
    

    stripes_seen = current_beat_ind(p, beats)
    
    if alternate:
        DIRECTION_X, DIRECTION_Y = ((-1)**(stripes_seen)) * DIRECTION_X, ((-1)**(stripes_seen)) * DIRECTION_Y

    overlay, extra = bar_overlay_transformer_adv(overlay, padding=padding, inter_bar_padding=0, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction)
    if extra is not None: 
        if stripes_seen == 0:
            time_taken = beats[0] * total_time
            start_prog = 0
        else:
            time_taken = (beats[stripes_seen] - beats[stripes_seen - 1]) * total_time
            start_prog = beats[stripes_seen - 1]

        tru_time_for_transition = min(time_taken, max_time)
        tru_total_progress_units = tru_time_for_transition/total_time

        current_prog = (p - start_prog)/tru_total_progress_units

        current_prog = min(max(0, current_prog),1)

        bright = (1-current_prog)**.5

        current_prog = current_prog ** .2
        current_prog = 1 - current_prog

        jitter = max_jitter * current_prog
        slide = min(max(current_prog, 0), 1) * 2


        extra = add_brightness(extra, bright)
        overlay = add_overlay(overlay, extra, (jitter * 2 * JITTER_X)  + (slide * DIRECTION_X), (jitter * 2 * JITTER_Y) + (slide * DIRECTION_Y))

        if horizontal:
            overlay = add_extra_motion_blur(overlay, BLUR * slide * .5, angle_deg=0)
        else:
            overlay = add_extra_motion_blur(overlay, BLUR * slide * .5, angle_deg=90)

    return add_overlay(frame, overlay, 0, 0)


def two_bar_rb_slide(*args, **kwargs):
    return _bars_extra_slide_adv_raw(*args, **kwargs, horizontal=False, NUM_BARS=2)

def five_bar_rb_slide(*args, **kwargs):
    return _bars_extra_slide_adv_raw(*args, **kwargs, horizontal=False, NUM_BARS=5)


def two_bar_rb_slide_horizontal(*args, **kwargs):
    return _bars_extra_slide_adv_raw(*args, **kwargs, horizontal=True, NUM_BARS=2)


def two_bar_rb_slide_alternate(*args, **kwargs):
    return _bars_extra_slide_adv_raw(*args, **kwargs, horizontal=False, NUM_BARS=2, alternate=True)


def two_bar_slide_frame(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=False, NUM_BARS=2, padding=.1)

def four_bar_slide_frame_alternate(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=False, NUM_BARS=4, padding=.1, alternate=True)


def two_bar_slide_frame_alternate(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=False, NUM_BARS=2, padding=.1, alternate=True)

def nine_bars_extra_slide(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=False, NUM_BARS=9)

def nine_bars_extra_slide_horizontal(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=True, NUM_BARS=9)

def five_bars_extra_slide(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=False, NUM_BARS=5)

def five_bars_extra_slide_horizontal(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=True, NUM_BARS=5)

def seven_bars_extra_slide_horizontal(*args, **kwargs):
    return _bars_extra_slide_raw(*args, **kwargs, horizontal=True, NUM_BARS=7)


def _bars_extra_raw(f1,f2,p,beats,total_time, block_progress, horizontal=False, NUM_BARS=5, bar_overlay_transformer_adv = transformers.bar_overlay_transformer_adv, add_brightness = transformers.add_brightness, max_jitter = 0.015, max_time = 0.3, pad = 0, inter_pad =0):
    #'NUM_BARS' Beats, perhaps as low as 120 BPM for this, but higher than that if possible
    
    if horizontal:
        direction = "horizontal"
        JITTER_Y = 1
        JITTER_X = 0
    else:
        direction = "vertical"
        JITTER_X = 1
        JITTER_Y = 0

    overlay = smartload(f2, block_progress[1])
    frame = smartload(f1, block_progress[0])
    

    stripes_seen = current_beat_ind(p, beats)
    
    overlay, extra = bar_overlay_transformer_adv(overlay, padding=pad, inter_bar_padding=inter_pad, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction)
    if extra is not None:
        if stripes_seen == 0:
            time_taken = beats[0] * total_time
            start_prog = 0
        else:
            time_taken = (beats[stripes_seen] - beats[stripes_seen - 1]) * total_time
            start_prog = beats[stripes_seen - 1]

        tru_time_for_transition = min(time_taken, max_time)
        tru_total_progress_units = tru_time_for_transition/total_time

        start_prog = beats[stripes_seen] - tru_total_progress_units
        if start_prog <= p <= beats[stripes_seen]:
            current_prog = (p - start_prog)/tru_total_progress_units

            current_prog = current_prog ** .2

            current_prog = 1 - current_prog

            jitter = max_jitter * current_prog
            bright = current_prog

            extra = add_brightness(extra, bright)
            overlay = add_overlay(overlay, extra, jitter * 2 * JITTER_X, jitter * 2 * JITTER_Y)

    return add_overlay(frame, overlay, 0, 0)

def nine_bars_extra(*args, **kwargs):
    return _bars_extra_raw(*args, **kwargs, horizontal=False, NUM_BARS=9)

def nine_bars_extra_horizontal(*args, **kwargs):
    return _bars_extra_raw(*args, **kwargs, horizontal=True, NUM_BARS=9)

def five_bars_extra(*args, **kwargs):
    return _bars_extra_raw(*args, **kwargs, horizontal=False, NUM_BARS=5)

def five_bars_extra_horizontal(*args, **kwargs):
    return _bars_extra_raw(*args, **kwargs, horizontal=True, NUM_BARS=5)


def four_bars_extra_diag(*args, **kwargs):
    return _bars_extra_raw(*args, **kwargs, horizontal=True, NUM_BARS=4, 
                           bar_overlay_transformer_adv=transformers.bar_overlay_transformer_adv_diag,
                           add_brightness=transformers.add_brightness                        
                        )

#Adaptive (3+ beats)
def bars_silo_diag(*args, **kwargs):
    beats = args[3] if len(args) >= 4 else kwargs["beats"]
    return _bars_extra_raw(*args, **kwargs, horizontal=True, NUM_BARS=len(beats), 
                           bar_overlay_transformer_adv=transformers.bar_overlay_transformer_adv_diag,
                           add_brightness=lambda x,y:transformers.add_brightness(x, 1),
                           max_jitter=0,
                           max_time=.05,
                           pad = .1,
                           inter_pad=0.05                        
                        )

def duck_rotate_breathe(f1,f2,p,beats,total_time, block_progress):
    #Takes in 1 beat

    frame = smartload(f1, block_progress[0]) if p <= beats[0] else smartload(f2, block_progress[1])
    still_point = .21 #Calculated using the warp transformer function
        
    if p <= beats[0]:
        #Calc the progress:
        current_prog = p/beats[0]
        turn_prog = still_point + (1 - still_point) * current_prog
    else:
        current_prog = (p - beats[0])/(1 - beats[0])
        turn_prog = still_point * current_prog
    
    return breathe_rotate_jiggle(turn_prog, frame)

#Idk just isnt working ;-;
def _bars_extra_jitter_raw(f1,f2,p,beats,total_time, block_progress, horizontal=False, NUM_BARS=5, padding = 0, alternate = False):
    #'NUM_BARS' Beats, if padding !=0, NUM_BARS + 1, perhaps as low as 120 BPM for this, but higher than that if possible
    max_jitter = 0.05
    max_time = 0.3

    if padding != 0 and p > beats[-1]:
        return smartload(f2, block_progress[1])

    if horizontal:
        direction = "horizontal"
        JITTER_Y = 1
        JITTER_X = 0

        DIRECTION_X = 0
        DIRECTION_Y = 0
    else:
        direction = "vertical"
        JITTER_X = 1
        JITTER_Y = 0

        DIRECTION_X = 0
        DIRECTION_Y = 0

    overlay = smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])

    stripes_seen = current_beat_ind(p, beats)
    
    if alternate:
        DIRECTION_X, DIRECTION_Y = ((-1)**(stripes_seen)) * DIRECTION_X, ((-1)**(stripes_seen)) * DIRECTION_Y

    overlay, extra = bar_overlay_transformer_adv(overlay, inter_bar_padding=0, num_bars=NUM_BARS, bars_visible=stripes_seen, direction=direction, padding=padding)
    TP = .5
    if extra is not None:
        if stripes_seen == 0:
            time_taken = beats[0] * total_time
            start_prog = 0
        else:
            time_taken = (beats[stripes_seen] - beats[stripes_seen - 1]) * total_time
            start_prog = beats[stripes_seen - 1]

        tru_time_for_transition = min(time_taken, max_time)
        tru_total_progress_units = tru_time_for_transition/total_time

        current_prog = (p - start_prog)/tru_total_progress_units

        current_prog = min(max(0, current_prog),1)
        current_prog = current_prog ** .3
        current_prog = 1 - current_prog

        jitter = max_jitter * progress_func.damped_jitter(current_prog, cycles_x=TP)[0]
        slide = min(max(current_prog, 0), 1) * 2


        extra = add_extra_motion_blur(extra, length_weighted=0.2 * progress_func.damped_jitter(current_prog, cycles_x=TP)[0])
        overlay = add_overlay(overlay, extra, (jitter * 2 * JITTER_X)  + (slide * DIRECTION_X), (jitter * 2 * JITTER_Y) + (slide * DIRECTION_Y))
        x0, y0, _, _ = add_overlay_bbox(overlay, extra, (jitter * 2 * JITTER_X)  + (slide * DIRECTION_X), (jitter * 2 * JITTER_Y) + (slide * DIRECTION_Y))
        dx1, dy1, dx2, dy2 = get_nontransparent_bbox(extra)
        x1, y1, x2, y2 = x0 + dx1, y0 + dy1, x0 + dx2, y0 + dy2

        h, w = frame.shape[:2]
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)

        return heat_streak_distortion_regional(add_overlay(frame, overlay, 0, 0), x1, y1, x2, y2, strength=30 * progress_func.damped_jitter(current_prog, cycles_x=TP)[0])

    return add_overlay(frame, overlay, 0, 0)


def bars_extra_jitter(*args, **kwargs):
    return _bars_extra_jitter_raw(*args, **kwargs)


#THIS WILL UPDATE ALL SLIDE_SHAKE_MULTI_POPULATIONS since this is a top level transition template
def _slide_shake_multi_population_raw(f1,f2,p,beats, total_time, block_progress, DIRECTION_VECTOR = [0,2]):
    #Beats -> 4 #beats_for_multi + 2 -> calibrated for 150 - 300 BPM, 
    #Random direction vector
    PAUSE = .4 #Possible cool effects if > 1
    PULSE_DURATION = 0.1 #Acutal duration = PULSE_DURATION * 2s
    beats_for_multi = 2 #


    beats_for_multi -= 1 #umm... nvm this, adjustment factor if you will
    if p <= beats[beats_for_multi + 2]:
        frame = smartload(f1, 1)
        if p <= beats[beats_for_multi + 0]:
            pop_prog = p/beats[beats_for_multi + 0]
        elif p <= beats[beats_for_multi + 1]:
            pop_prog = 1
        else:
            pop_prog = (beats[beats_for_multi + 2] - p)/(beats[beats_for_multi + 2] - beats[beats_for_multi + 1])
        frame = multiple_population(frame.copy(), progress_func.linear(pop_prog), scale=.3, num_copies=8)

    if p <= beats[beats_for_multi + 0]:
        pass
    elif beats[beats_for_multi + 0] < p < beats[beats_for_multi + 1]:
        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        slide_progress = ((p - beats[beats_for_multi + 0])/(beats[beats_for_multi + 1] - beats[beats_for_multi + 0]))
        slide_progress = progress_func.explosion(slide_progress) * PAUSE

        return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress)), False
    elif beats[beats_for_multi + 2] >= p >= beats[beats_for_multi + 1]:       
        slide_progress = ((p - beats[beats_for_multi + 1])/(beats[beats_for_multi + 2] - beats[beats_for_multi + 1]))
        slide_progress = progress_func.explosion(slide_progress)
        slide_progress = PAUSE + (1-PAUSE) * slide_progress

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)

        frame = add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    else:
        frame = smartload(f2, 0)

    distance_from_pulse = abs(p - beats[beats_for_multi + 2]) * total_time
    if distance_from_pulse <= PULSE_DURATION:
        pulse_progress = (PULSE_DURATION - distance_from_pulse) / (2 * PULSE_DURATION)
        pulse_progress = progress_func.explosion(pulse_progress)
        if p >= beats[beats_for_multi + 2]:
            p += .5

        f = heat_streak_distortion(frame, strength = pulse_progress * 3.5, direction="horizontal" if abs(DIRECTION_VECTOR[1]) > abs(DIRECTION_VECTOR[0]) else "vertical", streak_size_x=1, streak_size_y=15)
        return add_extra_motion_blur(f, 0.125 * (1 - (distance_from_pulse/PULSE_DURATION)), angle_deg=vector_to_angle_deg([x/2 for x in DIRECTION_VECTOR]) + 10), True
    
    return frame, False


@active_motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake_multi_population_up(*args, **kwargs):
    return _slide_shake_multi_population_raw(*args, **kwargs, DIRECTION_VECTOR=[0,2])

@active_motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake_multi_population_left(*args, **kwargs):
    return _slide_shake_multi_population_raw(*args, **kwargs, DIRECTION_VECTOR=[-2,0])

@active_motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake_multi_population_right(*args, **kwargs):
    return _slide_shake_multi_population_raw(*args, **kwargs, DIRECTION_VECTOR=[2,0])

@active_motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake_multi_population_down(*args, **kwargs):
    return _slide_shake_multi_population_raw(*args, **kwargs, DIRECTION_VECTOR=[0,-2])


#Better than above 4 because it is adaptive
@active_motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake_multi_population_smart(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return _slide_shake_multi_population_raw(*args, **kwargs, DIRECTION_VECTOR=DIRECTION_VECTOR)



def _slide_fade_raw(f1,f2,p,beats, total_time, block_progress, DIRECTION_VECTOR = [2,0], ZOOM = True):
    #Beats -> 1
    MIN_ZOOM = 0.05
    FADE_LENGTH = 0.1 #In seconds

    bf1, bf2 = block_progress #Even w/ alpha, supports playback because not blanket requires_rb

    frame = smartload(f1, bf1)
    if p >= 1 - (FADE_LENGTH/total_time):
        frame2 = smartload(f2, bf2)
        fade_prog = (p - (1 - (FADE_LENGTH/total_time)))/(FADE_LENGTH/total_time)
        fade_prog = progress_func.linear(fade_prog)

        frame = crossfade(frame, frame2, fade_prog)

    overlay = smartload(f2.replace('/RAW/', '/RB/'), bf2, alpha = True)

    if ZOOM:
        zoom = progress_func.sqrt(p)

        zoom = MIN_ZOOM + (1-MIN_ZOOM) * zoom
        overlay = resize_frame(overlay, zoom, zoom)

    slide_progress = p
    slide_progress = progress_func.explosion(slide_progress)

    overlay = resize_frame(overlay, slide_progress, slide_progress)
    return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    
def slide_fade_zoom(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    return _slide_fade_raw(*args, **kwargs, DIRECTION_VECTOR=[dx,dy], ZOOM = dz > 0)

def slide_fade(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    return _slide_fade_raw(*args, **kwargs, DIRECTION_VECTOR=[dx,dy], ZOOM=False)


def crossfade_cut(f1,f2,p,beats,total_time, block_progress):
    CROSS_FADE_DURATION = 0.2 #in seconds

    start = beats[0] - ((CROSS_FADE_DURATION/2)/total_time)
    end = beats[0] + ((CROSS_FADE_DURATION/2)/total_time)
    
    frame1 = smartload(f1, block_progress[0])
    frame2 = smartload(f2, block_progress[1])

    if p < start:
        return frame1
    if p > end:
        return frame2
    
    prog = (p - start)/(end - start)
    prog = progress_func.linear(prog)
    return crossfade_transform(frame1, frame2, prog)


def _slide_shake_squeeze_raw(f1,f2,p,beats, total_time, block_progress, DIRECTION_VECTOR = [2,0] ):
    #Beats -> 2
    beats = _generate_synthetic_beats(beats=beats)

    PAUSE = .6 #Possible cool effects if > 1
    BRIGHT_PULSE_DURATION = .1
    MAX_BRIGHT = 0.7
    SHAKE_DURATION = .1
    MAX_DISTORT = 1.2
    BLUR = .04

    if p < beats[0]:
        frame = smartload(f1, 1)

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)

        slide_progress = (p/beats[0])
        slide_progress = progress_func.explosion(slide_progress) * PAUSE

        overlay = resize_frame(overlay, slide_progress, slide_progress)
        return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    elif beats[1] >= p >= beats[0]:
        bright_total = min(beats[1] - beats[0],BRIGHT_PULSE_DURATION/total_time)
        bright_start = beats[1] - bright_total
        bright_prog = (p - bright_start)/(bright_total)

        frame = smartload(f1, 1)
        
        if bright_prog > 0:
            frame = fade_to_white(frame, bright_prog * MAX_BRIGHT)
        
        slide_progress = ((p - beats[0])/(beats[1] - beats[0]))
        slide_progress = progress_func.explosion(slide_progress)
        slide_progress = PAUSE + (1-PAUSE) * slide_progress

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        overlay = resize_frame(overlay, slide_progress, slide_progress)
        
        frame = add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    else:
        shake_total = min(1 - beats[1],SHAKE_DURATION/total_time)
        shake_start = 1 - shake_total
        shake_prog = (p - shake_start)/(shake_total)
        shake_prog = 1 - shake_prog

        frame = smartload(f2, 0)

        if 0 <= shake_prog <= 1:
            frame = crop_safe(resize_frame(frame.copy(), (MAX_DISTORT - 1) * shake_prog + 1,1), frame)
            frame = add_extra_motion_blur(frame, length_weighted=BLUR * shake_prog)


    return frame

def slide_shake_squeeze(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return _slide_shake_squeeze_raw(*args, **kwargs, DIRECTION_VECTOR=DIRECTION_VECTOR)


def _slide_shake_violent_raw(f1,f2,p,beats, total_time, block_progress, DIRECTION_VECTOR = [2,0] ):
    #Beats -> 2
    beats = _generate_synthetic_beats(beats)
    PAUSE = 1 #Possible cool effects if > 1
    PULSE_DURATION = 0.23 #Acutal duration = PULSE_DURATION
    DECAY = .8 #Half life = 0.69/DECAY progress units
    SHAKE_INTENSITY = .7
    BLUR = .3
    BRIGHT_PULSE = .12 #Duration in seconds
    EXTRA_JITTER = .085 #Strength of extra jitter, relative to minimum strength, i.e true intensity = e^(-DECAY) * EXTRA_JITTER * SHAKE_INTENSITY
    MAX_ZOOM = 1.12

    if p < beats[0]:
        frame = smartload(f1, 1)
        zoom = (MAX_ZOOM - 1) * progress_func.cubic(p/beats[1]) + 1
        frame = zoom_transform(frame, 1, zoom)

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        slide_progress = (p/beats[0])
        slide_progress = progress_func.explosion(slide_progress) * PAUSE

        return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    elif beats[1] >= p >= beats[0]:
        frame = smartload(f1, 1)
        zoom = (MAX_ZOOM - 1) * progress_func.cubic(p/beats[1]) + 1
        frame = zoom_transform(frame, 1, zoom)
        
        slide_progress = ((p - beats[0])/(beats[1] - beats[0]))
        slide_progress = progress_func.explosion(slide_progress)
        slide_progress = PAUSE + (1-PAUSE) * slide_progress

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)

        frame = add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    else:
        frame = smartload(f2, 0)


    if 0 <= (p - beats[1]) <= PULSE_DURATION/total_time:
        pulse_progress = (p - beats[1])/(PULSE_DURATION/total_time)
        #pulse_progress = progress_func.explosion(pulse_progress)
        decay_factor = math.exp(-DECAY * pulse_progress)
        dx,dy = list(progress_func.parametric_curve_from_seed_scaled(pulse_progress, f1 + f2, bias = [ij/2 for ij in DIRECTION_VECTOR], bias_strength = decay_factor))
        
        dx,dy = dx * SHAKE_INTENSITY * decay_factor, dy * SHAKE_INTENSITY * decay_factor
        
        #transformers.shake_transform(frame, pulse_progress,)
        frame =  transformers.raw_shake(frame, direction_vector=(dx,dy))
        frame = add_extra_motion_blur(frame, length_weighted=BLUR * decay_factor, angle_deg=vector_to_angle_deg((dx,dy)))

    else:
        decay_factor = math.exp(-DECAY) * EXTRA_JITTER
        dx,dy = list(progress_func.parametric_curve_from_seed_scaled(p, f1 + f2, bias = [ij/2 for ij in DIRECTION_VECTOR], bias_strength = decay_factor))
        dx,dy = dx * SHAKE_INTENSITY * decay_factor, dy * SHAKE_INTENSITY * decay_factor
        frame =  transformers.raw_shake(frame, direction_vector=(dx,dy))
        
    frame = transformers.zoom_transform(frame, 1, 1.05/(1-(SHAKE_INTENSITY * decay_factor))) #1/(1-strength) -> strength is max values in dx/dy

    if BRIGHT_PULSE/total_time >= p - beats[1] > 0:
        brightness = (p - beats[1])/(BRIGHT_PULSE/total_time)

        brightness = 1 - brightness**1.5

        frame = add_brightness(frame, brightness)

    return frame

def slide_shake_violent(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return _slide_shake_violent_raw(*args, **kwargs, DIRECTION_VECTOR=DIRECTION_VECTOR)

def _slide_shake_raw(f1,f2,p,beats, total_time, block_progress, DIRECTION_VECTOR = [2,0] ):
    #Beats -> 1
    beats = _generate_synthetic_beats(beats)
    PAUSE = .7 #Possible cool effects if > 1
    PULSE_DURATION = 0.07 #Acutal duration = PULSE_DURATION * 2s


    if p < beats[0]:
        frame = smartload(f1, 1)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)
        slide_progress = (p/beats[0])
        slide_progress = progress_func.explosion(slide_progress) * PAUSE

        return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    elif beats[1] >= p >= beats[0]:
        frame = smartload(f1, 1)
        
        slide_progress = ((p - beats[0])/(beats[1] - beats[0]))
        slide_progress = progress_func.explosion(slide_progress)
        slide_progress = PAUSE + (1-PAUSE) * slide_progress

        overlay = smartload(f2.replace('/RAW/', '/RB/'), 0, alpha = True)

        frame = add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    else:
        frame = smartload(f2, 0)

    distance_from_pulse = abs(p - beats[1]) * total_time
    if distance_from_pulse <= PULSE_DURATION:
        pulse_progress = (PULSE_DURATION - distance_from_pulse) / (2 * PULSE_DURATION)
        pulse_progress = progress_func.explosion(pulse_progress)
        if p >= beats[0]:
            p += .5

        frame2 = shake_transform(frame, pulse_progress, intensity=.15, randomness=.05, angular_intensity=10)

        return add_extra_motion_blur(frame2, 0.1 * pulse_progress)
    
    return frame

@motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake(*args, **kwargs):
    return _slide_shake_raw(*args, **kwargs, DIRECTION_VECTOR=[2,0])

@motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake2(*args, **kwargs):
    return _slide_shake_raw(*args, **kwargs, DIRECTION_VECTOR=[2,2])

@motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake3(*args, **kwargs):
    return _slide_shake_raw(*args, **kwargs, DIRECTION_VECTOR=[-2,0])

@motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shake4(*args, **kwargs):
    return _slide_shake_raw(*args, **kwargs, DIRECTION_VECTOR=[0,-2])

@motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=0.01)
def slide_shakex_smart(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return _slide_shake_raw(*args, **kwargs, DIRECTION_VECTOR=DIRECTION_VECTOR)


def trip_bars(f1,f2,p, beats, total_time, block_progress):
    #3 Beats
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2, block_progress[1])
    
    stripes_seen = current_beat_ind(p, beats) + 1
    if stripes_seen > 3:
        return overlay

    overlay, _ = bar_overlay_transformer_adv(overlay, padding=.1, inter_bar_padding=.05, num_bars=3, bars_visible=stripes_seen, direction="vertical")

    return add_overlay(frame, overlay, 0, 0)

def nine_bars(f1,f2,p,beats,total_time, block_progress):
    #9 Beats, only looks good with atleast 660 BPM or more (yes 660)
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2, block_progress[1])
    
    stripes_seen = current_beat_ind(p, beats) + 1
    if stripes_seen > 9:
        return overlay

    overlay, _ = bar_overlay_transformer_adv(overlay, padding=.07, inter_bar_padding=0, num_bars=9, bars_visible=stripes_seen, direction="vertical")

    return add_overlay(frame, overlay, 0, 0)


#quad silhoutte functions -> consolidated in quad_duo_silhoutte_cut_smart
def quad_silhouette_cut_white(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 4 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """       
    color = (255,255,255)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)

    if p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)


    if p < beats[0]:
        overlay2 = silhouette(overlay2, color)
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[3]:
        return add_overlay(frame, overlay, 0, 0)

def quad_silhouette_cut_white_zoom(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 4 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (255,255,255)
    MAX_ZOOM = 1.1

    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    frame = zoom_transform(frame, p/beats[-1],MAX_ZOOM)

    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)

    if p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

    if p < beats[0]:
        overlay2 = silhouette(overlay2, color)
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[3]:
        return add_overlay(frame, overlay, 0, 0)

def quad_silhouette_cut_black(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 4 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (0,0,0)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)

    if p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)


    if p < beats[0]:
        overlay2 = silhouette(overlay2, color)
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[3]:
        return add_overlay(frame, overlay, 0, 0)

def quad_silhouette_cut_black_zoom(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 4 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (0,0,0)
    MAX_ZOOM = 1.1

    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    frame = zoom_transform(frame, p/beats[-1],MAX_ZOOM)

    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)

    if p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

    if p < beats[0]:
        overlay2 = silhouette(overlay2, color)
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[3]:
        return add_overlay(frame, overlay, 0, 0)


#duo silhoutte functions -> consolidated in quad_duo_silhoutte_cut_smart
def double_silhouette_cut_white(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 2 beat
    """
    beats = _onebeat_back_beatsynth(beats, 1, 0)
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        overlay2 = silhouette(overlay, (255,255,255), False) #Color, Inverse
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        return add_overlay(frame, overlay, 0, 0)
    else:
        return smartload(f2, block_progress[1])

def double_silhouette_cut_black(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 2 beat
    """
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        overlay2 = silhouette(overlay, (0,0,0), False) #Color, Inverse
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        return add_overlay(frame, overlay, 0, 0)
    else:
        return smartload(f2, block_progress[1])

def double_silhouette_cut_white_zoom(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 2 beat
    """
    beats = _onebeat_back_beatsynth(beats, 1, 0)
    MAX_ZOOM = 1.1

    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        frame = zoom_transform(frame, p/beats[-1],MAX_ZOOM)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        overlay2 = silhouette(overlay, (255,255,255), False) #Color, Inverse
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        frame = smartload(f1, block_progress[0])
        frame = zoom_transform(frame, p/beats[-1],MAX_ZOOM)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        return add_overlay(frame, overlay, 0, 0)
    else:
        return smartload(f2, block_progress[1])

def double_silhouette_cut_black_zoom(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 2 beat
    """
    beats = _onebeat_back_beatsynth(beats, 1, 0)
    MAX_ZOOM = 1.1

    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        frame = zoom_transform(frame, p/beats[-1],MAX_ZOOM)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        overlay2 = silhouette(overlay, (255,255,255), False) #Color, Inverse
        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        frame = smartload(f1, block_progress[0])
        frame = zoom_transform(frame, p/beats[-1],MAX_ZOOM)
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        return add_overlay(frame, overlay, 0, 0)
    else:
        return smartload(f2, block_progress[1])

#Rerouting function (based on f2 -> decide which one of the 4, based on weights (and the shape/model ofc))
quad_duo_reroute = dict()
quad_duo_color = "white" #Can also be "black", figure out a way to create it based on /MEDIA/ -> mostly dark/mostly bright then take opposite
zoom_probability = 0.25
told_about_quad = False
def quad_duo_silhoutte_cut_smart(f1,f2,p,beats,total_time,block_progress):
    #Only needs 1 beat
    #Improvements: Re-factor functions to make it a single consolidated function constructor which can be created and assigned to quad_duo_reroute
    #Improvements: Automatically decide if white/black silohoutte looks better

    global quad_duo_reroute
    if f2 in quad_duo_reroute:
        my_func, should_duo = quad_duo_reroute[f2]
        beats = _onebeat_back_beatsynth(beats, 1 if should_duo else 3, 0)
        return my_func(f1,f2,p, beats, total_time, block_progress)

    global told_about_quad
    if not told_about_quad:
        print("\n\nLooks like you are using a silohoutte function -> If you want to always spawn the images in 2 parts you can place the first portion in /RB_HIGHLIGHT/ directory adjacent to the /RAW/ directory with the same name...")
        print("Tip: It almost always looks better this way, cars? add in the headlight only! Person? Maybe something they are holding!")
        safe_input("You can click enter to continue...\n\n")
        told_about_quad = True

    should_zoom = random.random() <= zoom_probability
    if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
        should_duo = False
    else:
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        should_duo = len_split_overlay(overlay) == 1
    
    if should_duo and should_zoom and quad_duo_color == "white":
        my_func = double_silhouette_cut_white_zoom
    if should_duo and should_zoom and quad_duo_color == "black":
        my_func = double_silhouette_cut_black_zoom
    if not should_duo and should_zoom and quad_duo_color == "white":
        my_func = quad_silhouette_cut_white_zoom
    if not should_duo and should_zoom and quad_duo_color == "black":
        my_func = quad_silhouette_cut_black_zoom
    if should_duo and not should_zoom and quad_duo_color == "white":
        my_func = double_silhouette_cut_white
    if should_duo and not should_zoom and quad_duo_color == "black":
        my_func = double_silhouette_cut_black
    if not should_duo and not should_zoom and quad_duo_color == "white":
        my_func = quad_silhouette_cut_white
    if  not should_duo and not should_zoom and quad_duo_color == "black":
        my_func = quad_silhouette_cut_black

    quad_duo_reroute[f2] = (my_func, should_duo) 

    return quad_duo_silhoutte_cut_smart(f1,f2,p,beats,total_time,block_progress) #Re-call to reuse cached code


#trip and mono -> consolidated in tri_mono_silhoutte_cut_smart
def trip_silhouette_cut_white(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 3 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (255,255,255)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[2]:
        return add_overlay(frame, overlay, 0, 0)

def trip_silhouette_cut_black(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 3 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (0, 0, 0)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[1]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[2]:
        return add_overlay(frame, overlay, 0, 0)

def trip_silhouette_cut_white_load(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 3 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (255,255,255)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, silhouette(overlay2, color), 0, 0)
    elif p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        return add_overlay(frame, overlay, 0, 0)

def trip_silhouette_cut_black_load(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 3 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (0,0,0)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, silhouette(overlay2, color), 0, 0)
    elif p < beats[1]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, overlay2, 0, 0)
    elif p < beats[2]:
        return add_overlay(frame, overlay, 0, 0)

def trip_silhouette_cut_white_ds(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 3 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (255,255,255)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, silhouette(overlay2, color), 0, 0)
    elif p < beats[1]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[2]:
        return add_overlay(frame, overlay, 0, 0)

def trip_silhouette_cut_black_ds(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 3 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    color = (0,0,0)
    if p >= beats[-1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, silhouette(overlay2, color), 0, 0)
    elif p < beats[1]:
        overlay = silhouette(overlay, color)
        return add_overlay(frame, overlay, 0, 0)
    elif p < beats[2]:
        return add_overlay(frame, overlay, 0, 0)

def silhouette_cut_white(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 1 beat
    """
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        overlay2 = silhouette(overlay, (255,255,255), False) #Color, Inverse
        return add_overlay(frame, overlay2, 0, 0)    
    else:
        return smartload(f2, block_progress[1])

def silhouette_cut_black(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 1 beat
    """
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        overlay2 = silhouette(overlay, (0,0,0), False) #Color, Inverse
        return add_overlay(frame, overlay2, 0, 0)    
    else:
        return smartload(f2, block_progress[1])


tri_mono_reroute = dict()
tri_mono_color = "white" #Can also be "black", figure out a way to create it based on /MEDIA/ -> mostly dark/mostly bright then take opposite

def tri_mono_silhoutte_cut_smart(f1,f2,p,beats,total_time,block_progress):
    #Only needs 1 beat
    #Improvements: Re-factor functions to make it a single consolidated function constructor which can be created and assigned to quad_duo_reroute
    #Improvements: Automatically decide if white/black silohoutte looks better

    global tri_mono_reroute

    if f2 in tri_mono_reroute:
        my_func, should_mono = tri_mono_reroute[f2]
        beats = beats if should_mono else _onebeat_back_beatsynth(beats, 2, 0)
        return my_func(f1,f2,p, beats, total_time, block_progress)

    #Maintains same intro input so as to not repeat...
    global told_about_quad
    if not told_about_quad:
        print("\n\nLooks like you are using a silohoutte function -> If you want to always spawn the images in 2 parts you can place the first portion in /RB_HIGHLIGHT/ directory adjacent to the /RAW/ directory with the same name...")
        print("Tip: It almost always looks better this way, cars? add in the headlight only! Person? Maybe something they are holding!")
        safe_input("You can click enter to continue...\n\n")
        told_about_quad = True


    if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
        should_mono = False
    else:
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        should_mono = len_split_overlay(overlay) == 1
    
    if should_mono:
        my_func = silhouette_cut_white if tri_mono_color == "white" else silhouette_cut_black
    else:
        if tri_mono_color == "white":
            my_func = random.choices(
                [trip_silhouette_cut_white, trip_silhouette_cut_white_load, trip_silhouette_cut_white_ds],
                weights=[0.25,0.25,0.5], # Because I like ds versions more lol
                k = 1
            )[0]
        elif tri_mono_color == "black":
            my_func = random.choices(
                [trip_silhouette_cut_black, trip_silhouette_cut_black_load, trip_silhouette_cut_black_ds],
                weights=[0.25,0.25,0.5], # Because I like ds versions more lol
                k = 1
            )[0]
        
    tri_mono_reroute[f2] = (my_func, should_mono) 

    return tri_mono_silhoutte_cut_smart(f1,f2,p,beats,total_time,block_progress) #Re-call to reuse cached code


#Consolidated via partial_image_quick_cut
def double_image_quick_cut(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 2 beat,
    Add the file in "MEDIA/RB_HIGHLIGHT/nameoffile.png" to create a custom main object overlay
    """
    if p >= beats[1]:
        return smartload(f2, block_progress[1])
    
    frame = smartload(f1, block_progress[0])
    overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)


    if p < beats[0]:
        if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
            overlay2 = smartload(f2.replace('/RAW/', '/RB_HIGHLIGHT/'), block_progress[1], alpha = True)
        else:
            overlay2, _ = split_overlay(overlay)

        return add_overlay(frame, overlay2, 0, 0)
    
    return add_overlay(frame, overlay, 0, 0)

def single_image_quick_cut(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 1 beat
    """
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        return add_overlay(frame, overlay, 0, 0)    
    elif p >= beats[0]:
        return smartload(f2, block_progress[1])

partial_image_quick_cut_reroute = dict()
def partial_image_quick_cut(f1,f2,p,beats,total_time,block_progress):
    #Only needs 1 beat
    #Improvements: Re-factor functions to make it a single consolidated function constructor which can be created and assigned to quad_duo_reroute
    #Improvements: Automatically decide if white/black silohoutte looks better

    global partial_image_quick_cut_reroute

    if f2 in partial_image_quick_cut_reroute:
        my_func, should_mono = partial_image_quick_cut_reroute[f2]
        beats = beats if should_mono else _onebeat_back_beatsynth(beats, 1, 0)
        return my_func(f1,f2,p, beats, total_time, block_progress)

    #Maintains same intro input so as to not repeat...
    global told_about_quad
    if not told_about_quad:
        print("\n\nLooks like you are using a silohoutte function -> If you want to always spawn the images in 2 parts you can place the first portion in /RB_HIGHLIGHT/ directory adjacent to the /RAW/ directory with the same name...")
        print("Tip: It almost always looks better this way, cars? add in the headlight only! Person? Maybe something they are holding!")
        safe_input("You can click enter to continue...\n\n")
        told_about_quad = True


    if os.path.exists(f2.replace('/RAW/', '/RB_HIGHLIGHT/').replace(".mp4", "_front.png")):
        should_mono = False
    else:
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        should_mono = len_split_overlay(overlay) == 1
    
    my_func = single_image_quick_cut if should_mono else double_image_quick_cut        
        
    partial_image_quick_cut_reroute[f2] = (my_func, should_mono) 

    return partial_image_quick_cut(f1,f2,p,beats,total_time,block_progress) #Re-call to reuse cached code

def zoom_out_ns_block(f1,f2,p,beats,total_time, block_progress):
    #2 beats, extend block progress, fast BPM -> >400BPM, ideally 600 BPM even
    beats = _onebeat_back_beatsynth([beats[0]], 1, 0) + _onebeat_back_beatsynth([beats[1]], 0, 1)
    MAX_ZOOM = 1.4 #For the internal frame
    OTHER_ZOOM = 1 #For the outer frame
    CROPS = [.3, .15, 0.05, 0] #Padding on each side
    SILO = None #None for no silohoutte

    if p < beats[0]:
        return smartload(f1, block_progress[0])
    
    frame = smartload(f2,block_progress[1])
    
    zoom = MAX_ZOOM - ((p - beats[0])/(1 - beats[0])) * (MAX_ZOOM - 1)
    frame = zoom_transform(frame, 1, zoom)
    
    cropped, _ = bar_overlay_transformer_adv(frame, CROPS[current_beat_ind(p, beats) - 1], num_bars=1, inter_bar_padding=0, bars_visible=1, direction="vertical")
    
    if p < beats[-1]:
        zoom = OTHER_ZOOM - ((p - beats[0])/(beats[-1] - beats[0])) * (OTHER_ZOOM - 1)
        cropped = resize_frame(cropped, zoom, zoom)

    
    if SILO is not None and current_beat_ind(p, beats) == 1:
        cropped = silhouette(cropped, color=SILO)

    return add_overlay(smartload(f1, block_progress[0]), cropped, 0, 0)

#legacy, use the reverse tool in the builtin editor
def reverse_linear(f1,f2,p,beats,total_time, block_progress):
    if p > 0.5:
        return smartload(f2, block_progress[1] if block_progress[1] < .5 else 1 - block_progress[1])
    else: 
        return smartload(f1, block_progress[0] if block_progress[0] < .5 else 1 - block_progress[0])

def zoom_out_silohoutte_block_white(f1,f2,p,beats,total_time, block_progress):
    #2 beats, extend block progress, fast BPM -> >400BPM, ideally 600 BPM even
    beats = _onebeat_back_beatsynth([beats[0]], 1, 0) + _onebeat_back_beatsynth([beats[1]], 0, 1)
    MAX_ZOOM = 1.4 #For the internal frame
    OTHER_ZOOM = 1 #For the outer frame
    CROPS = [.3, .15, 0.05, 0] #Padding on each side
    SILO = (255,255,255) #None for no silohoutte

    if p < beats[0]:
        return smartload(f1, block_progress[0])
    
    frame = smartload(f2,block_progress[1])
    
    zoom = MAX_ZOOM - ((p - beats[0])/(1 - beats[0])) * (MAX_ZOOM - 1)
    frame = zoom_transform(frame, 1, zoom)
    
    cropped, _ = bar_overlay_transformer_adv(frame, CROPS[current_beat_ind(p, beats) - 1], num_bars=1, inter_bar_padding=0, bars_visible=1, direction="vertical")
    
    if p < beats[-1]:
        zoom = OTHER_ZOOM - ((p - beats[0])/(beats[-1] - beats[0])) * (OTHER_ZOOM - 1)
        cropped = resize_frame(cropped, zoom, zoom)

    
    if SILO is not None and current_beat_ind(p, beats) == 1:
        cropped = silhouette(cropped, color=SILO)

    return add_overlay(smartload(f1, block_progress[0]), cropped, 0, 0)

def zoom_out_silohoutte_block_black(f1,f2,p,beats,total_time, block_progress):
    #2 beats, extend block progress, fast BPM -> >400BPM, ideally 600 BPM even
    beats = _onebeat_back_beatsynth([beats[0]], 1, 0) + _onebeat_back_beatsynth([beats[1]], 0, 1)
    MAX_ZOOM = 1.4 #For the internal frame
    OTHER_ZOOM = 1 #For the outer frame
    CROPS = [.3, .15, 0.05, 0] #Padding on each side
    SILO = (0,0,0) #None for no silohoutte

    if p < beats[0]:
        return smartload(f1, block_progress[0])
    
    frame = smartload(f2,block_progress[1])
    
    zoom = MAX_ZOOM - ((p - beats[0])/(1 - beats[0])) * (MAX_ZOOM - 1)
    frame = zoom_transform(frame, 1, zoom)
    
    cropped, _ = bar_overlay_transformer_adv(frame, CROPS[current_beat_ind(p, beats) - 1], num_bars=1, inter_bar_padding=0, bars_visible=1, direction="vertical")
    
    if p < beats[-1]:
        zoom = OTHER_ZOOM - ((p - beats[0])/(beats[-1] - beats[0])) * (OTHER_ZOOM - 1)
        cropped = resize_frame(cropped, zoom, zoom)

    
    if SILO is not None and current_beat_ind(p, beats) == 1:
        cropped = silhouette(cropped, color=SILO)

    return add_overlay(smartload(f1, block_progress[0]), cropped, 0, 0)

def skin_effect_cut(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 1 beat
    """
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        
        prog = progress_func.explosion(p/beats[0])
        
        overlay2 = skin_effect(overlay, p = prog)
        return add_overlay(frame, overlay2, 0, 0)    

    elif p >= beats[0]:
        return smartload(f2, block_progress[1])


def image_quick_cut_bright2(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 2 beat -> Higher BPM wanted
    """
    beats = _onebeat_back_beatsynth(beats, 1, 0)
    if p < beats[1]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        bright = (1 - min(1, p/beats[0])) * .5
        
        #prog = progress_func.explosion(p/beats[0])
        #overlay2 = skin_effect(overlay, p = prog)

        overlay2 = add_brightness(overlay, bright)
        return add_overlay(frame, overlay2, 0, 0)    
    elif p >= beats[1]:
        return smartload(f2, block_progress[1])

def image_quick_cut_bright(f1,f2,p, beats, total_time, block_progress):
    """
    Takes in 1 beat
    """
    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        bright = (1 - min(1, p/beats[0])) * .5
        
        #prog = progress_func.explosion(p/beats[0])
        #overlay2 = skin_effect(overlay, p = prog)

        overlay2 = add_brightness(overlay, bright)
        return add_overlay(frame, overlay2, 0, 0)    
    elif p >= beats[0]:
        return smartload(f2, block_progress[1])


def background_quick_change_bright(f1,f2,p,beats,total_time, block_progress):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    #Any number of beats, BOTH TRANSPARENT
    root_dir = os.path.dirname(f1)
    time_per_img = .1 #in seconds
    
    #Basically arbitary numbers, but they work 
    progress_till_last_frame = .9
    fade_out_at = .48 * progress_till_last_frame 
    stop_fade_out_at = .67 * progress_till_last_frame
    move_in_at = .91 * progress_till_last_frame
    
    brightness_pulse_dist = 0.15/total_time #in progress units on each side, i.e seconds/total_time
    
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]


    motion_blur_time_diff = time_per_img/20
    motion_blur_total_frames = 5

    if p > progress_till_last_frame:
        final_frame = smartload(f2, block_progress[1])
    else:
        file_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
        ]

        file_paths = [f for f in file_paths if f.endswith(".png")]
        file_paths = sorted([f for f in file_paths if f.lower() not in {f1.lower(),f2.lower()}])
        
        overlay = smartload(f1.replace('/RAW/', '/RB/'), block_progress[0], alpha = True)

        if p > fade_out_at:
            fade_prog = (p-fade_out_at)/(stop_fade_out_at - fade_out_at)
            overlay = alpha_add(overlay, 1 - min(fade_prog, 1))
    

        current_time = p * total_time
        current_img_ind = int(current_time/time_per_img)
        current_img_path = file_paths[current_img_ind%len(file_paths)]
        current_frame = smartload(current_img_path, (current_time/time_per_img) - current_img_ind)
                
        my_frames = []
        for x in range(-motion_blur_total_frames//2 + 1, motion_blur_total_frames//2 + 1):    
            zoom = max(1, 1.5 - (((current_time + (x * motion_blur_time_diff))/time_per_img) - current_img_ind)*.5)
        
            t_frame = zoom_transform(current_frame, 1, zoom).copy()
            my_frames.append(t_frame)
            #my_frames.append(add_overlay(t_frame, skin_effect(overlay2, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog))
        
            my_frames.append(zoom_transform(current_frame, 1, zoom).copy())

        final_frame = add_overlay(motion_blur_temporal(my_frames), overlay, 0, 0)
    
    if progress_till_last_frame > p >= move_in_at:
        overlay2 = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha=True)
        move_prog = (p-move_in_at)/(progress_till_last_frame - move_in_at)
        move_prog = 1 - progress_func.explosion(move_prog)
        final_frame = add_overlay(final_frame, resize_frame(overlay2, 1 - move_prog, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog)    

    if abs(progress_till_last_frame - p) < brightness_pulse_dist:
        bright = abs(progress_till_last_frame - p)/brightness_pulse_dist
        bright = 1 - bright

        start_p = progress_till_last_frame - brightness_pulse_dist
        end_p = progress_till_last_frame + brightness_pulse_dist
        pulse = (p-start_p)/(end_p - start_p)
        
        
        final_frame = pulse_transform(final_frame, pulse, .7, .1)

        #final_frame = add_extra_motion_blur(final_frame, MAX_BLUR * blur, angle_deg=vector_to_angle_deg(DIRECTION_VECTOR),)
        final_frame = fade_to_white(final_frame, bright ** .5)

    return final_frame


def background_quick_change_dark(f1,f2,p,beats,total_time, block_progress):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    root_dir = os.path.dirname(f1)
    time_per_img = .1 #in seconds
    
    #Basically arbitary numbers 
    progress_till_last_frame = .95
    fade_out_at = .48 * progress_till_last_frame 
    stop_fade_out_at = .67 * progress_till_last_frame
    move_in_at = .96 * progress_till_last_frame
    
    brightness_pulse_dist = 0.15/total_time #in progress units on each side, i.e seconds/total_time
     
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]
    
    MAX_BLUR = 0.1

    motion_blur_time_diff = time_per_img/20
    motion_blur_total_frames = 5

    if p > progress_till_last_frame:
        final_frame = smartload(f2, block_progress[1])
    else:
        file_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
        ]

        file_paths = [f for f in file_paths if f.endswith(".png")]
        file_paths = sorted([f for f in file_paths if f.lower() not in {f1.lower(),f2.lower()}])
        
        overlay = smartload(f1.replace('/RAW/', '/RB/'), block_progress[0], alpha = True)

        if p > fade_out_at:
            fade_prog = (p-fade_out_at)/(stop_fade_out_at - fade_out_at)
            overlay = alpha_add(overlay, 1 - min(fade_prog, 1))
    

        current_time = p * total_time
        current_img_ind = int(current_time/time_per_img)
        current_img_path = file_paths[current_img_ind%len(file_paths)]
        current_frame = smartload(current_img_path, (current_time/time_per_img) - current_img_ind)
                
        my_frames = []
        for x in range(-motion_blur_total_frames//2 + 1, motion_blur_total_frames//2 + 1):    
            zoom = max(1, 1.5 - (((current_time + (x * motion_blur_time_diff))/time_per_img) - current_img_ind)*.5)
        
            t_frame = zoom_transform(current_frame, 1, zoom).copy()
            my_frames.append(t_frame)
            #my_frames.append(add_overlay(t_frame, skin_effect(overlay2, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog))
        
            my_frames.append(zoom_transform(current_frame, 1, zoom).copy())

        final_frame = add_overlay(motion_blur_temporal(my_frames), overlay, 0, 0)
    
    if progress_till_last_frame > p >= move_in_at:
        overlay2 = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha=True)
        move_prog = (p-move_in_at)/(progress_till_last_frame - move_in_at)
        move_prog = 1 - progress_func.explosion(move_prog)
        final_frame = add_overlay(final_frame, resize_frame(overlay2, 1 - move_prog, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog)    

    if abs(progress_till_last_frame - p) < brightness_pulse_dist:
        bright = abs(progress_till_last_frame - p)/brightness_pulse_dist
        bright = 1 - bright

        start_p = progress_till_last_frame - brightness_pulse_dist
        end_p = progress_till_last_frame + brightness_pulse_dist
        pulse = (p-start_p)/(end_p - start_p)
        
        
        final_frame = pulse_transform(final_frame, pulse, .7, .1)

        #final_frame = add_extra_motion_blur(final_frame, MAX_BLUR * blur, angle_deg=vector_to_angle_deg(DIRECTION_VECTOR),)
        final_frame = fade_to_black(final_frame, bright ** .5)

    return final_frame


def background_quick_change_shake(f1,f2,p,beats,total_time, block_progress):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    #Ideally these should last around 7-10s, 8.2 was the sample vid
    root_dir = os.path.dirname(f1)
    time_per_img = .1 #in seconds
    
    #Basically arbitary numbers based on the sample vid
    progress_till_last_frame = .95
    fade_out_at = .48 * progress_till_last_frame 
    stop_fade_out_at = .67 * progress_till_last_frame
    move_in_at = .96 * progress_till_last_frame
    
    brightness_pulse_dist = 0.1/total_time #in progress units on each side, i.e seconds/total_time

    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]


    MAX_BLUR = 0.1

    motion_blur_time_diff = time_per_img/20
    motion_blur_total_frames = 5

    if p > progress_till_last_frame:
        final_frame = smartload(f2, block_progress[1])
    else:
        file_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
        ]

        file_paths = [f for f in file_paths if f.endswith(".png")]
        file_paths = sorted([f for f in file_paths if f.lower() not in {f1.lower(),f2.lower()}])
        
        overlay = smartload(f1.replace('/RAW/', '/RB/'), block_progress[0], alpha = True)

        if p > fade_out_at:
            fade_prog = (p-fade_out_at)/(stop_fade_out_at - fade_out_at)
            overlay = alpha_add(overlay, 1 - min(fade_prog, 1))
    

        current_time = p * total_time
        current_img_ind = int(current_time/time_per_img)
        current_img_path = file_paths[current_img_ind%len(file_paths)]
        current_frame = smartload(current_img_path, (current_time/time_per_img) - current_img_ind)
                
        my_frames = []
        for x in range(-motion_blur_total_frames//2 + 1, motion_blur_total_frames//2 + 1):    
            zoom = max(1, 1.5 - (((current_time + (x * motion_blur_time_diff))/time_per_img) - current_img_ind)*.5)
        
            t_frame = zoom_transform(current_frame, 1, zoom).copy()
            my_frames.append(t_frame)
            #my_frames.append(add_overlay(t_frame, skin_effect(overlay2, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog))
        
            my_frames.append(zoom_transform(current_frame, 1, zoom).copy())

        final_frame = add_overlay(motion_blur_temporal(my_frames), overlay, 0, 0)
    
    if progress_till_last_frame > p >= move_in_at:
        overlay2 = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha=True)
        move_prog = (p-move_in_at)/(progress_till_last_frame - move_in_at)
        move_prog = 1 - progress_func.explosion(move_prog)
        final_frame = add_overlay(final_frame, resize_frame(overlay2, 1 - move_prog, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog)    

    if abs(progress_till_last_frame - p) < brightness_pulse_dist:
        blur = abs(progress_till_last_frame - p)/brightness_pulse_dist
        blur = 1 - blur

        start_p = progress_till_last_frame - brightness_pulse_dist
        end_p = progress_till_last_frame + brightness_pulse_dist
        pulse = (p-start_p)/(end_p - start_p)
        
        
        final_frame = pulse_transform(final_frame, pulse, .7, .1)

        final_frame = add_extra_motion_blur(final_frame, MAX_BLUR * blur, angle_deg=vector_to_angle_deg(DIRECTION_VECTOR),)


    return final_frame

def linear_to_background_quick_change(f1,f2,p,beats,total_time, block_progress):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    root_dir = os.path.dirname(f1)
    time_per_img = .1 #in seconds
    progress_till_last_frame = .99
    max_zoom = 1.2 #Can be higher as well, goes from max_zoom to 1

    motion_blur_time_diff = time_per_img/20
    motion_blur_total_frames = 5
    
    prog_per_img = time_per_img/total_time
    
    if p < beats[0]:
        return smartload(f1, block_progress[0])
    if p > progress_till_last_frame:
        return smartload(f2, block_progress[1])
    else:
        file_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
        ]

        file_paths = [f for f in file_paths if f.endswith(".png")]
        file_paths = sorted([f for f in file_paths if f.lower() not in {f1.lower(),f2.lower()}])

        current_prog = (p - beats[0])/(progress_till_last_frame - beats[0])

        zoom = 1 + (max_zoom - 1) * ((1 - current_prog)**.5)    
        overlay2 = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha=True)
        overlay2 = resize_frame(overlay2, zoom, zoom)
         

        current_time = p * total_time
        current_img_ind = int(current_time/time_per_img)

        if p + prog_per_img >= progress_till_last_frame:
            current_frame = smartload(f2, block_progress[1])
        else:
            current_img_path = file_paths[current_img_ind%len(file_paths)]
            current_frame = smartload(current_img_path, (current_time/time_per_img) - current_img_ind) #Although this block progress looks wonky, pretty sure its correct once you factorise everything
             
        my_frames = []
        for x in range(-motion_blur_total_frames//2 + 1, motion_blur_total_frames//2 + 1):    
            zoom = max(1, 1.5 - (((current_time + (x * motion_blur_time_diff))/time_per_img) - current_img_ind)*.5)
        
            t_frame = zoom_transform(current_frame, 1, zoom).copy()
            my_frames.append(t_frame)
            #my_frames.append(add_overlay(t_frame, skin_effect(overlay2, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog))
        
            my_frames.append(zoom_transform(current_frame, 1, zoom).copy())

        

        final_frame = add_overlay(motion_blur_temporal(my_frames), overlay2, 0, 0)
        
    return final_frame


GIVEN_PRIORITY_INFO = False
def priority_background_quick_change_shake(f1,f2,p,beats,total_time, block_progress):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    global GIVEN_PRIORITY_INFO
    root_dir = os.path.dirname(f1).replace('RAW', 'BQC')
    
    if not GIVEN_PRIORITY_INFO:
        print("\n\nYo! Add pics to /MEDIA/BQC/ if you want those for the rapid background change effect. If it can't find, it will default to /MEDIA/RAW/")
        safe_input("Click Enter to continue...\n\n")
        GIVEN_PRIORITY_INFO = True

        os.makedirs(root_dir, exist_ok= True)

    time_per_img = .1 #in seconds

    # Arbitary numbers 
    progress_till_last_frame = .95
    fade_out_at = .48 * progress_till_last_frame 
    stop_fade_out_at = .67 * progress_till_last_frame
    move_in_at = .96 * progress_till_last_frame

    
    brightness_pulse_dist = 0.1/total_time #in progress units on each side, i.e seconds/total_time

    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    MAX_BLUR = 0.1

    motion_blur_time_diff = time_per_img/20
    motion_blur_total_frames = 5

    if p > progress_till_last_frame:
        final_frame = smartload(f2, block_progress[1])
    else:
        file_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        ]

        if len(file_paths) == 0:
            root_dir = root_dir.replace('/BQC', '/RAW')
            file_paths = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if os.path.isfile(os.path.join(root_dir, f))
            ]



        file_paths = [f for f in file_paths if f.endswith(".png")]
        file_paths = sorted([f for f in file_paths if f.lower() not in {f1.lower(),f2.lower()}])
        
        overlay = smartload(f1.replace('/RAW/', '/RB/'), block_progress[0], alpha = True)

        if p > fade_out_at:
            fade_prog = (p-fade_out_at)/(stop_fade_out_at - fade_out_at)
            overlay = alpha_add(overlay, 1 - min(fade_prog, 1))
    

        current_time = p * total_time
        current_img_ind = int(current_time/time_per_img)
        current_img_path = file_paths[current_img_ind%len(file_paths)]
        current_frame = smartload(current_img_path, (current_time/time_per_img) - current_img_ind)
                
        my_frames = []
        for x in range(-motion_blur_total_frames//2 + 1, motion_blur_total_frames//2 + 1):    
            zoom = max(1, 1.5 - (((current_time + (x * motion_blur_time_diff))/time_per_img) - current_img_ind)*.5)
        
            t_frame = zoom_transform(current_frame, 1, zoom).copy()
            my_frames.append(t_frame)
            #my_frames.append(add_overlay(t_frame, skin_effect(overlay2, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog))
        
            my_frames.append(zoom_transform(current_frame, 1, zoom).copy())

        final_frame = add_overlay(motion_blur_temporal(my_frames), overlay, 0, 0)
    
    if progress_till_last_frame > p >= move_in_at:
        overlay2 = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha=True)
        move_prog = (p-move_in_at)/(progress_till_last_frame - move_in_at)
        move_prog = 1 - progress_func.explosion(move_prog)
        final_frame = add_overlay(final_frame, resize_frame(overlay2, 1 - move_prog, 1 - move_prog), DIRECTION_VECTOR[0]*move_prog, DIRECTION_VECTOR[1]*move_prog)    

    if abs(progress_till_last_frame - p) < brightness_pulse_dist:
        blur = abs(progress_till_last_frame - p)/brightness_pulse_dist
        blur = 1 - blur

        start_p = progress_till_last_frame - brightness_pulse_dist
        end_p = progress_till_last_frame + brightness_pulse_dist
        pulse = (p-start_p)/(end_p - start_p)
        
        
        final_frame = pulse_transform(final_frame, pulse, .7, .1)

        final_frame = add_extra_motion_blur(final_frame, MAX_BLUR * blur, angle_deg=vector_to_angle_deg(DIRECTION_VECTOR),)


    return final_frame

#Add in a second one -> 0.06 second fade in, 0.03 second stay, 0.06 second fade out | bleed effect
#Also add in a quick change zooming outward but no overlay...

GIVEN_P_INFO = set()

def _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress, p_in_trans, p_out_trans, zoom_prog_func = lambda x:x, p_disable = False, raw_folder = None):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    time_per_img = .12 #in seconds
    MAX_ZOOM = 3
    if p_disable:
        p_in_prog_units = 0
        p_out_prog_units = 0
        p_end = 0
    else:
        p_in_prog_units = min(0.2, 0.3/total_time)
        p_out_prog_units = min(0.2, 0.3/total_time)
        p_end = min(0.25, 0.15/total_time)
    

    if p > 1 - p_end:
        return smartload(f2, block_progress[1]), False
    
    flag2 = 0
    if p < p_in_prog_units:
        p_new = p/p_in_prog_units
        flag2 = 1
    if p > 1 - (p_out_prog_units + p_end):
        flag2 = 2
        p_new = (p - (1-(p_out_prog_units + p_end)))/p_out_prog_units
    
    p = (p - p_in_prog_units)/(1 - p_in_prog_units - p_out_prog_units - p_end)
    tt = (1 - p_in_prog_units - p_out_prog_units - p_end) * total_time
    num_of_images = int(tt//time_per_img)
    if num_of_images <= 0:
        raise Exception("Uh oh, time period is too small...")
    time_per_img = tt/num_of_images
    prog_per_img = time_per_img/total_time

    global GIVEN_P_INFO
    random.seed(f1 + f2)
    idnumber = random.randint(1_000_000,10_000_000 - 1)
    root_dir = os.path.dirname(f1).replace('RAW', f'BQC{idnumber}')
    
    flag1 = False
    if idnumber not in GIVEN_P_INFO:
        print(f"\n\nYo! Add pics to MEDIA/BQC{idnumber}/ if you want those for the rapid background change effect. If it can't find, it will default to /MEDIA/RAW/")
        print(f"files: {f1} -> {f2}")
        os.makedirs(root_dir, exist_ok= True)
        print(root_dir)
        safe_input("Click Enter to continue...\n\n")
        GIVEN_P_INFO.add(idnumber)
        flag1 = True

    #Get media paths
    file_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
    ]


    file_paths = [filepath for filepath in file_paths if filepath.lower()]

    if len(file_paths) == 0:
        root_dir = root_dir.replace(f'BQC{idnumber}', 'RAW') if raw_folder is None else raw_folder
        file_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        ]

        if flag1:
            print("No media in BQC, defaulting to /RAW/")

    file_paths = sorted(file_paths)
    random.shuffle(file_paths)

    if flag2 == 1:
        img_path = file_paths[(num_of_images - 1)%len(file_paths)]
        return p_out_trans(img_path,f1, f2, p_new), False #out cv2 BGR

    if flag2 == 2:
        img_path = file_paths[0]
        return p_in_trans(img_path, f1,f2,p_new),False # -> out cv2 BGR
    

    img_num = int(p//prog_per_img)
    img_path = file_paths[img_num%len(file_paths)]
    p_zoom = (p - (img_num * prog_per_img))/prog_per_img

    zoom = (zoom_prog_func(p_zoom) * (MAX_ZOOM - 1)) + 1
    
    frame = smartload(img_path, p_zoom)
    frame = transformers.zoom_transform(frame, 1, zoom)
    

    frame = transformers.radial_blur(frame, 50)

    #overlay = smartload(f2.replace('/RAW/','/RB/'), block_progress[1], alpha=True)
    overlay = smartload(f2.replace('/RAW/','/RB/'), p if p_disable else 1 - p, alpha=True, make_alpha = True)

    return add_overlay(frame, overlay, 0, 0), False

def _white_splash(backf1, overf1, f2, p):
    bp = (0,0,0) #Replace with effective bps if applying to videos
    frame1 = add_overlay(smartload(backf1, bp[0]), smartload(overf1, bp[1], alpha=True),0,0)
    frame2 = smartload(f2, bp[0])

    p = p/2 # temporary fix... probably not temporary tho ;)

    if p < 0.5:
        frame = frame1
        fade_prog = p/0.5
    else:
        frame = frame2
        fade_prog = (1 - p)/0.5

    return fade_to_white(frame, progress_func.linear(fade_prog))


def _anti_silo_out(backf1, overf1, f2, p):
    bp = (0,0,0) #Replace with effective bps if applying to videos
    
    frame1 = smartload(backf1, bp[0])
    
    frame = solid_color_frame(frame1, (255,255,255))
    
    if p < 0.5:
        overlay_raw = smartload(overf1, bp[1], alpha=True)
        frame = add_overlay(frame, overlay_raw, 0, 0)
    
    return frame

def _shake_brr_simple(backf1, overf1, f2, p):
    bp = (0,0,0) #Replace with effective bps if applying to videos
    frame1 = add_overlay(smartload(backf1, bp[0]), smartload(overf1, bp[1], alpha=True),0,0)
    frame2 = smartload(f2, bp[0])

    return shake_brr22(frame1, frame2, p, [0.5], 0.3, (1,0))


BQC_ZOOM_BLUR_SETTINGS = (3,.1/40)
#Zooming in, white cuts
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def bqc1(f1,f2,p,beats,total_time, block_progress):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_white_splash,
                            p_out_trans=_white_splash,
                            zoom_prog_func=lambda x:x**1.5
    )
    
#Zooming out, white cuts
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def bqc2(f1,f2,p,beats,total_time, block_progress):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_white_splash,
                            p_out_trans=_white_splash,
                            zoom_prog_func=lambda x:(1 - x)**1.5
    )
    

#Zooming in, shake cut
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def bqc3(f1,f2,p,beats,total_time, block_progress):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_shake_brr_simple,
                            p_out_trans=_shake_brr_simple,
                            zoom_prog_func=lambda x:x**1.5
    )
    
#Zooming out, shake cut
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def bqc4(f1,f2,p,beats,total_time, block_progress):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_shake_brr_simple,
                            p_out_trans=_shake_brr_simple,
                            zoom_prog_func=lambda x:(1 - x)**1.5
    )
    


def _background_qc_adv_blend(f1,f2,p,beats,total_time, block_progress, p_in_trans, p_out_trans, zoom_prog_func = lambda x:x, p_disable = False, raw_folder = None):
    #Get root directory of f1 -> Flexible on bpm since only depends on time_per_img, maybe create a couple different references, adjusting time per thing
    time_split = (0.06,0.03,0.06)
    time_per_img = sum(time_split) #in seconds

    if p_disable:
        p_in_prog_units = 0
        p_out_prog_units = 0
        p_end = 0
    else:
        p_in_prog_units = min(0.2, 0.3/total_time)
        p_out_prog_units = min(0.2, 0.3/total_time)
        p_end = min(0.25, 0.15/total_time)


    if p > 1 - p_end:
        return smartload(f2, block_progress[1])

    flag2 = 0
    if p < p_in_prog_units:
        p_new = p/p_in_prog_units
        flag2 = 1
    if p > 1 - (p_out_prog_units + p_end):
        flag2 = 2
        p_new = (p - (1-(p_out_prog_units + p_end)))/p_out_prog_units
    
    p = (p - p_in_prog_units)/(1 - p_in_prog_units - p_out_prog_units - p_end)
    tt = (1 - p_in_prog_units - p_out_prog_units - p_end) * total_time
    num_of_images = int(tt//time_per_img)
    if num_of_images <= 0:
        raise Exception("Uh oh, time period is too small...")
       
    time_per_img = tt/num_of_images
    prog_per_img = time_per_img/total_time

    tsr = [time_split[i]/sum(time_split) for i in range(len(time_split))]
    tsr = [sum(tsr[:i+1]) for i in range(len(tsr))]

    global GIVEN_P_INFO
    random.seed(f1 + f2)
    idnumber = random.randint(1_000_000,10_000_000 - 1)
    root_dir = os.path.dirname(f1).replace('RAW', f'BQC{idnumber}')
    
    flag1 = False
    if idnumber not in GIVEN_P_INFO:
        print(f"\n\nYo! Add pics to MEDIA/BQC{idnumber}/ if you want those for the rapid background change effect. If it can't find, it will default to /MEDIA/RAW/")
        print(f"files: {f1} -> {f2}")
        os.makedirs(root_dir, exist_ok= True)
        print(root_dir)
        safe_input("Click Enter to continue...\n\n")
        GIVEN_P_INFO.add(idnumber)
        flag1 = True

    #Get media paths
    file_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
    ]


    file_paths = [filepath for filepath in file_paths if filepath.lower()]

    if len(file_paths) == 0:
        root_dir = root_dir.replace(f'BQC{idnumber}', 'RAW') if raw_folder is None else raw_folder
        file_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        ]
        if flag1:
            print("No media in BQC, defaulting to /RAW/")

    file_paths = sorted(file_paths)
    random.shuffle(file_paths)

    if flag2 == 1:
        img_path = file_paths[(num_of_images - 1)%len(file_paths)]
        return p_out_trans(img_path,f1, f2, p_new) #out cv2 BGR

    if flag2 == 2:
        img_path = file_paths[0]
        return p_in_trans(img_path, f1,f2,p_new) # -> out cv2 BGR
    

    img_num = int(p//prog_per_img)
    img_path = file_paths[img_num%len(file_paths)]

    if img_num == 0:
        img_path0 = str(img_path)
    else:
        img_path0 = file_paths[(img_num%len(file_paths)) - 1]
    if img_num == num_of_images - 1:
        img_path1 = str(img_path)
    else:
        img_path1 = file_paths[(img_num + 1)%len(file_paths)]
    

    p_zoom = (p - (img_num * prog_per_img))/prog_per_img

    
    frame = smartload(img_path, 0.75) #Constant video position
    if p_zoom <= tsr[0]:
        frame0 = smartload(img_path0, 1)
        frame = transformers.crossfade(frame0, frame, p_zoom/tsr[0])
    elif p_zoom >= tsr[2]:
        frame1 = smartload(img_path1, 0)
        frame = transformers.crossfade(frame, frame1, (p_zoom - tsr[2])/(1 - tsr[2]))

    

    #overlay = smartload(f2.replace('/RAW/','/RB/'), block_progress[1], alpha=True, make_alpha = False)
    overlay = smartload(f2.replace('/RAW/','/RB/'), p if p_disable else 1 - p, alpha=True, make_alpha = True)

    return add_overlay(frame, overlay, 0, 0)

def bqc5(f1,f2,p,beats,total_time, block_progress):
    return _background_qc_adv_blend(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_shake_brr_simple,
                            p_out_trans=_shake_brr_simple,
                            zoom_prog_func=lambda x:(1 - x)**1.5
    )


#Zooming in, white cuts
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def gbqc1(f1,f2,p,beats,total_time, block_progress, raw_folder):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_white_splash,
                            p_out_trans=_white_splash,
                            zoom_prog_func=lambda x:(1 - x)**1.5,
                            p_disable=True,
                            raw_folder=raw_folder
    )
    
#Zooming out, white cuts
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def gbqc2(f1,f2,p,beats,total_time, block_progress, raw_folder):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_white_splash,
                            p_out_trans=_white_splash,
                            zoom_prog_func=lambda x:(1 - x)**1.5,
                            p_disable=True,
                            raw_folder=raw_folder
    )
    

#Zooming in, shake cut
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def gbqc3(f1,f2,p,beats,total_time, block_progress, raw_folder):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_shake_brr_simple,
                            p_out_trans=_shake_brr_simple,
                            zoom_prog_func=lambda x:(1 - x)**1.5,
                            p_disable=True,
                            raw_folder=raw_folder
    )
    
#Zooming out, shake cut
@active_motion_blur(*BQC_ZOOM_BLUR_SETTINGS)
def gbqc4(f1,f2,p,beats,total_time, block_progress, raw_folder):
    return _background_qc_adv_zoom(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_shake_brr_simple,
                            p_out_trans=_shake_brr_simple,
                            zoom_prog_func=lambda x:(1 - x)**1.5,
                            p_disable=True,
                            raw_folder=raw_folder
    )
    

def gbqc5(f1,f2,p,beats,total_time, block_progress, raw_folder):
    return _background_qc_adv_blend(f1,f2,p,beats,total_time, block_progress,
                            p_in_trans=_shake_brr_simple,
                            p_out_trans=_shake_brr_simple,
                            zoom_prog_func=lambda x:(1 - x)**1.5,
                            p_disable=True,
                            raw_folder=raw_folder
    )

# -- 
def bars_cut_quick_rb(f1,f2,p, beats, total_time, block_progress):
    #Any number of beats
    beats = _generate_synthetic_beats(beats, True)
    if p < beats[0]:
        return smartload(f1, block_progress[0])
    elif p >= beats[0]:
        total_beats = len(beats) - 1
        current_beats = len([x for x in beats if p >= x]) - 1
        current_size = .6 + progress_func.square(current_beats / total_beats) * .4

        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha=True)
        overlay2 = bar_mask(overlay, total_beats, current_beats, direction=0)
        overlay3 = resize_frame(overlay2, current_size, current_size)
        
        return add_overlay(frame, overlay3, 0, 0)



#Already smart
def slide_zoom_shake(f1,f2,p,beats, total_time, block_progress):
    #Beats -> 2
    PULSE_DURATION = 0.12 #Acutal duration = PULSE_DURATION * 2s
    
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    MAX_BRIGHTNESS = 1
    MAX_SIZE_INCREASE = 1.3
    MAX_ROTATION = 5 #In degrees

    if p < beats[0]:
        frame = smartload(f1, block_progress[0])
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)
        slide_progress = (p/beats[0])
        slide_progress = progress_func.explosion(slide_progress)

        
        p_extra = p/beats[1]
        p_extra = progress_func.square(p_extra)
        frame = zoom_rotate_transform(zoom_transform(frame, p_extra, MAX_SIZE_INCREASE), p_extra, theta = MAX_ROTATION)

        return add_overlay(frame, overlay, DIRECTION_VECTOR[0] * (1-slide_progress), DIRECTION_VECTOR[1] * (1-slide_progress))
    elif beats[1] >= p >= beats[0]:
        frame = smartload(f1, block_progress[0])

        p_extra = p/beats[1]
        p_extra = progress_func.square(p_extra)
        frame = zoom_rotate_transform(zoom_transform(frame, p_extra, MAX_SIZE_INCREASE), p_extra, theta = MAX_ROTATION)
        
        overlay = smartload(f2.replace('/RAW/', '/RB/'), block_progress[1], alpha = True)

        frame = add_overlay(frame, overlay, 0, 0)
    else:
        frame = smartload(f2, block_progress[1])

    distance_from_pulse = abs(p - beats[1]) * total_time
    if distance_from_pulse <= PULSE_DURATION:
        pulse_progress = (PULSE_DURATION - distance_from_pulse) / (2 * PULSE_DURATION)
        pulse_progress = progress_func.explosion(pulse_progress)
        if p >= beats[0]:
            p += .5
        
        #Switch with some actual shake animation -> GOT TO CREATE A GREAT CAMERA SHAKE ANIMATION
        return shake_transform(add_brightness(frame, MAX_BRIGHTNESS * (1 - (distance_from_pulse/PULSE_DURATION))), pulse_progress,
                               intensity=.5,        # now a fraction of frame size
                 randomness=0.2,
                 bounciness=0.8,
                 direction=(x/2 for x in DIRECTION_VECTOR), # bias direction in [-1,1]
                 angular_intensity=15)
    return frame

@motion_blur(motion_blur_total_frames=3, motion_blur_time_diff=.007)
def shake_cut(f1, f2, p, beats, total_time, block_progress):
    #Beats -> 1
    BEAT_LENGTH = .2
    direction_vector = (1,0)

    frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    beat_start = beats[0] - BEAT_LENGTH
    beats_end = beats[0] + BEAT_LENGTH

    if beat_start <= p <= beats_end:
        beats_progress = (p - beat_start) / (beats_end - beat_start)
        #return shake_transform(frame, beats_progress, intensity=.5)
        return camera_shake(add_extra_motion_blur(frame, 0.06 * abs(p - beats[0]/BEAT_LENGTH), angle_deg=vector_to_angle_deg(direction_vector)), beats_progress, intensity_factor=0.05, direction_vector=direction_vector)
    else:
        return frame

def _shake_blur_raw(f1,f2,p,beats,total_time, block_progress, DIRECTION_VECTOR = [0, 1]):
    #Beats -> 1
    BEAT_LENGTH = .1/total_time #Time in seconds to do it
    
    frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    beat_start = beats[0] - BEAT_LENGTH/2
    beats_end = beats[0] + BEAT_LENGTH/2

    if beat_start <= p <= beats_end:
        beats_progress = (p - beat_start) / (beats_end - beat_start)
        return shake_transform(frame, beats_progress,
                 intensity=0.5,        # now a fraction of frame size
                 randomness=0.02,
                 bounciness=0.9,
                 direction=(DIRECTION_VECTOR[0], DIRECTION_VECTOR[1]), # bias direction in [-1,1]
                 angular_intensity=15), True
    else:
        return frame, False

@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=9)
def shake_blur_1(*args, **kwargs):
    return _shake_blur_raw(*args, **kwargs, DIRECTION_VECTOR=[0.1, 1])

@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=9)
def shake_blur_2(*args, **kwargs):
    return _shake_blur_raw(*args, **kwargs, DIRECTION_VECTOR=[1, .9])

@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=9)
def shake_blur_3(*args, **kwargs):
    return _shake_blur_raw(*args, **kwargs, DIRECTION_VECTOR=[1, 0.1])

@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=9)
def shake_blur_4(*args, **kwargs):
    return _shake_blur_raw(*args, **kwargs, DIRECTION_VECTOR=[-.9, 1])

@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=9)
def shake_blurx_smart(*args, **kwargs):
    f2 = args[1] if len(args) >= 2 else kwargs["f2"]
    dx,dy,dz = _get_best_slide_dv(f2)
    DIRECTION_VECTOR = [dx,dy]

    return _shake_blur_raw(*args, **kwargs, DIRECTION_VECTOR=DIRECTION_VECTOR)


@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=5)
def zoom_shake_blur(f1,f2,p,beats,total_time, block_progress):
    #Honestly looks great in any BPM, for faster paced, it creates a nice bouncy effect, and for slower ones should create a nice cinematic pan
    #Beats -> 1
    BEAT_LENGTH = .1/total_time #Time in seconds to do it
    max_zoom_pan = 1.04
    max_zoom_extra = 1.2

    frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    beat_start = beats[0] - BEAT_LENGTH/2
    beats_end = beats[0] + BEAT_LENGTH/2

    if p < beat_start:
        zoom = (beat_start - p)/beat_start #0 at max zoom, 1 at least zoom
    elif p > beats_end:
        zoom = (p - beats_end)/(1 - beats_end) #0 at max zoom, 1 at least zoom
    else:
        zoom = 1
        
    zoom = progress_func.square(zoom)
    zoom = max_zoom_pan - (max_zoom_pan - 1)*zoom

    if beat_start <= p <= beats_end:
        beats_progress = (p - beat_start) / (beats_end - beat_start)
        
        zoom = abs(p - beats[0])/(BEAT_LENGTH/2) #Get progress, where 0 is max zoom
        
        zoom = progress_func.square(zoom)
        zoom = max_zoom_extra - (max_zoom_extra - max_zoom_pan)*zoom
        frame = zoom_transform(frame, 1, zoom)

        return shake_transform(frame, beats_progress,
                 intensity=0.5,        # now a fraction of frame size
                 randomness=0.02,
                 bounciness=0.9,
                 direction=(1.0, 1.0), # bias direction in [-1,1]
                 angular_intensity=15), True
    else:
        frame = zoom_transform(frame, 1, zoom)
        return frame, False


@active_motion_blur(motion_blur_time_diff=.01, motion_blur_total_frames=5)
def zoom_shake_blur_slide(f1,f2,p,beats,total_time, block_progress):
    #Honestly looks great in any BPM, for faster paced, it creates a nice bouncy effect, and for slower ones should create a nice cinematic pan
    #Beats -> 1
    BEAT_LENGTH = .1/total_time #Time in seconds to do it
    max_zoom_pan = 1.04
    max_zoom_extra = 1.2

    frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    beat_start = beats[0] - BEAT_LENGTH/2
    beats_end = beats[0] + BEAT_LENGTH/2

    if p < beat_start:
        zoom = (beat_start - p)/beat_start #0 at max zoom, 1 at least zoom
    elif p > beats_end:
        zoom = (p - beats_end)/(1 - beats_end) #0 at max zoom, 1 at least zoom
    else:
        zoom = 1
        
    zoom = progress_func.square(zoom)
    zoom = max_zoom_pan - (max_zoom_pan - 1)*zoom

    if beat_start <= p <= beats_end:
        beats_progress = (p - beat_start) / (beats_end - beat_start)
        
        if p < beats[0]:
            overlay = smartload(f2, block_progress[1])
            slide_prog_loc = ((p - beats[0])/(beat_start - beats[0]))
            frame = add_overlay(frame, overlay, 2 * slide_prog_loc, 2 * slide_prog_loc)

        zoom = abs(p - beats[0])/(BEAT_LENGTH/2) #Get progress, where 0 is max zoom
        
        zoom = progress_func.square(zoom)
        zoom = max_zoom_extra - (max_zoom_extra - max_zoom_pan)*zoom
        frame = zoom_transform(frame, 1, zoom)

        return shake_transform(frame, beats_progress,
                 intensity=0.5,        # now a fraction of frame size
                 randomness=0.02,
                 bounciness=0.9,
                 direction=(0, 0), # bias direction in [-1,1]
                 angular_intensity=15), True
    else:
        frame = zoom_transform(frame, 1, zoom)
        return frame, False

def brightness_shake_cut(f1, f2, p, beats, total_time, block_progress):
    #Beats -> 1
    BEAT_TIME = 0.28 #Beat time in seconds
    MAX_BRIGHT = .6
    MAX_WARP = .4 #Extra size increase (total size = 1 + MAX_WARP)
    BEAT_LENGTH = (BEAT_TIME/total_time)
    
    direction_vector = (.1, 1) #DO NOT TOUCH

    frame = linear_cut(f1, f2, p, beats, total_time, block_progress)
    beat_start = max(0, beats[0] - BEAT_LENGTH) #Clip
    beats_end = min(1, beats[0] + BEAT_LENGTH)


    #bright_start, bright_end = bright_start - (beats_end - beat_start)/FAC, bright_start + (beats_end - beat_start)/FAC
    beat_start = beats[0]

    if beat_start <= p <= beats_end:
        beats_progress = (p - beat_start) / (beats_end - beat_start)
        
        #return shake_transform(frame, beats_progress, intensity=.5)
        frame = camera_shake(add_extra_motion_blur(frame, 0.05 * abs(p - beats[0]/BEAT_LENGTH), angle_deg=vector_to_angle_deg(direction_vector)), beats_progress, intensity_factor=0.05, direction_vector=direction_vector)
        
    if beats[0] <= p:
        #prog = max(min((abs((p - beats[0])/(bright_start - bright_end))), 1), 0)
        prog = (p - beats[0])/(1 - beats[0])
        brightness = progress_func.sharp_curve_off(prog)
        
        warp_prog = 1 - prog**1.5
        overlay = resize_frame(frame, 1 + warp_prog * MAX_WARP * direction_vector[0] , 1 + warp_prog *  MAX_WARP * direction_vector[1])
        frame = add_overlay(frame, overlay, warp_prog * MAX_WARP * direction_vector[0] * -1, warp_prog * MAX_WARP * direction_vector[1] * -1)

        
        frame = add_brightness(frame, brightness * MAX_BRIGHT)

    return frame

def fade_to_black_cut_intro(f1,f2,p, beats, total_time, block_progress):
    #Takes in as many beats as needed
    beats = [beats[len(beats)//2]] #Take middle one
    frame = linear_cut(f1,f2,p, beats, total_time, block_progress)
    if abs(p - beats[0]) < .1:
        pulse_progress = ((.1 - abs(p - beats[0])) / .1)*.5
        return fade_to_black(frame, pulse_progress)
    return frame

