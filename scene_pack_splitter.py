import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk, ImageDraw
import subprocess
import os
import atexit
import time
import ctypes
import cv2
import math
import settings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import numpy as np


TEMP_AUDIO_PATH = "_temp_audio.wav"
TEMP_VIDEO_PATH = "_temp_video.mp4"
TEMP_PROXY_PATH = "_temp_proxy.mp4"
PREVIEW_MAX_SIZE = 580


def cleanup():
    for path in [TEMP_AUDIO_PATH, TEMP_VIDEO_PATH, TEMP_PROXY_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass
atexit.register(cleanup)


def make_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


#I think something like this exists in pygame, couldn't find it though...
class AudioPlayer:
    def __init__(self):
        try:
            pygame.mixer.init()
            self.initialized = True
        except pygame.error as e:
            print(f"Pygame Mixer Error: {e}")
            self.initialized = False
    def play(self, filepath):
        if not self.initialized or not os.path.exists(filepath): return
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
        except pygame.error as e: print(f"Could not play audio: {e}")
    def pause(self):
        if self.initialized: pygame.mixer.music.pause()
    def unpause(self):
        if self.initialized: pygame.mixer.music.unpause()
    def stop(self):
        if self.initialized:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    def is_busy(self):
        if not self.initialized: return False
        return pygame.mixer.music.get_busy()
    def close(self):
        if self.initialized: pygame.mixer.quit()

class SceneSplitter(tk.Frame):
    def __init__(self, master=None, default_filename_func = lambda x: f"{x}_trimmed.mp4", default_dir = None, default_pic = lambda x: f"{x}_screenshot.png", default_aspect_ratio = None):
        super().__init__(master)

        self.default_filename_func = default_filename_func
        self.default_dir = default_dir
        self.default_pic_func = default_pic
        self.default_ar = default_aspect_ratio
        self.prev_render_zoom_frame = None

        self.master.geometry("900x750")
        self.master.title("R=Reset Frame | Z=Undo Keyframe | L=Lock | K=Add Keyframe")
        self.pack(fill=tk.BOTH, expand=True)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.video_path, self.cap = None, None
        self.duration_seconds, self.start_time, self.end_time, self.preview_time = 0.0, 0.0, 0.0, 0.0
        self.is_playing, self.is_paused = False, False
        self.playback_start_system_time, self.playback_start_video_time = 0, 0
        self.preview_image_tk, self.timeline_dragging_handle = None, None
        self.fps = None
        self.image_display_size = (0, 0)
        
        self.fps = 0
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        self.preview_frame_num = 0


        self.crop_state = {'cx': 0.5, 'cy': 0.5, 'w': 1.0, 'h': 1.0, 'angle': 0.0}
        self.keyframes = {}
        self.keyframe_history = []
        self.crop_locked = False
        self.crop_dragging_handle = None
        self.crop_dragging_start_state = {}

        self.audio_player = AudioPlayer()
        if not self.audio_player.initialized:
            messagebox.showwarning("Audio Warning", "Could not initialize Pygame mixer. Audio playback will be disabled.")

        self.create_widgets()
        self.setup_key_bindings()

    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.load_button = tk.Button(top_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))
        self.video_label = tk.Label(top_frame, text="No video loaded.", bg="#ddd", anchor="w")
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.preview_container = tk.Frame(self, bg="black")
        self.preview_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.preview_container.grid_propagate(False)
        self.preview_frame = tk.Label(self.preview_container, bg="black")
        self.preview_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        self.preview_frame.bind("<ButtonPress-1>", self.on_crop_press)
        self.preview_frame.bind("<B1-Motion>", self.on_crop_drag)
        self.preview_frame.bind("<ButtonRelease-1>", self.on_crop_release)

        self.timeline_canvas = tk.Canvas(self, height=50, bg="#f0f0f0", cursor="arrow", bd=0, highlightthickness=0)
        self.timeline_canvas.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.timeline_canvas.bind("<Configure>", lambda e: self.draw_timeline())
        self.timeline_canvas.bind("<ButtonPress-1>", self.on_timeline_press)
        self.timeline_canvas.bind("<B1-Motion>", self.on_timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.on_timeline_release)

        time_frame = tk.Frame(self)
        time_frame.grid(row=3, column=0, sticky="ew", padx=10)
        self.start_time_label = tk.Label(time_frame, text="Start: 00:00:00.00")
        self.start_time_label.pack(side=tk.LEFT, padx=5)
        self.preview_time_label = tk.Label(time_frame, text="Cursor: 00:00:00.00", font=("Segoe UI", 10, "bold"))
        self.preview_time_label.pack(side=tk.LEFT, expand=True)
        self.end_time_label = tk.Label(time_frame, text="End: 00:00:00.00")
        self.end_time_label.pack(side=tk.RIGHT, padx=5)

        action_frame = tk.Frame(self)
        action_frame.grid(row=4, column=0, pady=(15, 10))
        self.play_pause_button = tk.Button(action_frame, text="▶ Play", command=self.toggle_play_pause, state=tk.DISABLED, width=10)
        self.play_pause_button.pack(side=tk.LEFT, padx=10)
        self.lock_button = tk.Button(action_frame, text="Lock Crop", command=self.toggle_crop_lock_event, state=tk.DISABLED, width=12)
        self.lock_button.pack(side=tk.LEFT, padx=10)
        self.trim_button = tk.Button(action_frame, text="Trim and Save Video", command=self.trim_video, state=tk.DISABLED, bg="#cce5ff")
        self.trim_button.pack(side=tk.LEFT, padx=10)

    def load_video(self):
        self.stop_playback()
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if not path: return
        if self.cap: self.cap.release()
        self.video_path, self.cap = path, cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file with OpenCV.")
            return
        self.video_label.config(text=os.path.basename(path))
        

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.start_frame = 0
        self.end_frame = self.total_frames
        self.preview_frame_num = 0

        self.start_time = 0.0
        self.end_time = self.duration_seconds
        self.preview_time = 0.0


        self.play_pause_button.config(state=tk.NORMAL); self.trim_button.config(state=tk.NORMAL)
        self.og_dimensions = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.lock_button.config(state=tk.NORMAL)
        self.full_reset()
        self.update_idletasks()
        self.update_all_ui()
        self.update_frame_preview(0)

    def load_video(self):
        self.stop_playback()
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if not path: return
        

        if self.cap: self.cap.release()
        
        self.video_path = path # Keeps track of the ORIGINAL High-Res file
        self.video_label.config(text=os.path.basename(path))


        temp_cap = cv2.VideoCapture(self.video_path)
        if not temp_cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return
        
        w = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        temp_cap.release()

        # Decide whether to use proxy (low res for faster previews)
        self.using_proxy = False
        load_path = self.video_path

        if max(w, h) > PREVIEW_MAX_SIZE:
            self.video_label.config(text=f"{os.path.basename(path)} (Generating Proxy...)")
            self.master.update() # Force UI update to show "Generating Proxy"
            
            scale_cmd = f"-vf scale={PREVIEW_MAX_SIZE}:-2" if w >= h else f"-vf scale=-2:{PREVIEW_MAX_SIZE}"
            
            cmd = (f'ffmpeg -i "{self.video_path}" {scale_cmd} '
                   f'-sws_flags neighbor -c:v libx264 -preset ultrafast -crf 28 -an -y "{TEMP_PROXY_PATH}"')
            
            try:
                subprocess.run(cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                load_path = TEMP_PROXY_PATH
                self.using_proxy = True
                self.video_label.config(text=f"{os.path.basename(path)} (Proxy Active)")
            except subprocess.CalledProcessError:
                print("Proxy generation failed. Using original.")
                load_path = self.video_path
                self.video_label.config(text=os.path.basename(path))

        self.cap = cv2.VideoCapture(load_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video stream.")
            return



        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.start_frame = 0
        self.end_frame = self.total_frames
        self.preview_frame_num = 0

        self.start_time = 0.0
        self.end_time = self.duration_seconds
        self.preview_time = 0.0


        self.play_pause_button.config(state=tk.NORMAL); self.trim_button.config(state=tk.NORMAL)


        self.og_dimensions = (int(w), int(h)) 
        self.lock_button.config(state=tk.NORMAL)
        self.full_reset()
        self.update_idletasks()
        self.update_all_ui()
        self.update_frame_preview(0)

    def render_native_frame(self, frame, state, source_dims):
        """Renders a single frame by cropping, returning the native resolution."""
        source_w, source_h = source_dims

        center_x_abs = state['cx'] * source_w
        center_y_abs = state['cy'] * source_h
        crop_w_abs = int(state['w'] * source_w)
        crop_h_abs = int(state['h'] * source_h)

        M = cv2.getRotationMatrix2D((center_x_abs, center_y_abs), -state['angle'], 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (source_w, source_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        x1 = max(0, int(center_x_abs - crop_w_abs / 2))
        y1 = max(0, int(center_y_abs - crop_h_abs / 2))
        x2 = min(source_w, x1 + crop_w_abs)
        y2 = min(source_h, y1 + crop_h_abs)
        
        return rotated_frame[y1:y2, x1:x2]

    def update_frame_preview(self, frame_num):
        if not self.cap or not self.cap.isOpened(): return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            # Calculate current time for interpolation, based on the frame number, had to use frames instead of time for frame-accuracy when using arrow keys
            current_time_seconds = frame_num / self.fps if self.fps > 0 else 0

            container_w, container_h = self.preview_container.winfo_width(), self.preview_container.winfo_height()
            if container_w < 10 or container_h < 10: return
            h, w, _ = frame.shape; aspect_ratio = w / h
            new_w, new_h = container_w, container_h
            if (container_w / aspect_ratio) <= container_h: new_h = int(container_w / aspect_ratio)
            else: new_w = int(container_h * aspect_ratio)
            if new_w < 1 or new_h < 1: return
            
            self.image_display_size = (new_w, new_h)
            
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            if self.crop_dragging_handle:
                current_state = self.crop_state
            else:
                current_state = self.get_interpolated_state(current_time_seconds)
            
            final_image = self.draw_crop_overlay(img, current_state)
            self.preview_image_tk = ImageTk.PhotoImage(image=final_image)
            self.preview_frame.config(image=self.preview_image_tk)

    def trim_video(self):
        self.stop_playback()
        if self.start_time >= self.end_time:
            messagebox.showwarning("Invalid Range", "The start time must be before the end time.")
            return
        
        active_keyframes = {t: s for t, s in self.keyframes.items() if self.start_time <= t <= self.end_time}
        

        self.trim_video_opencv()
        return #Atm ffmpeg shouldn't be used, isn't working well...
        
        if len(active_keyframes) < 2:
            self.trim_video_ffmpeg()
        else:
            self.trim_video_opencv()
    
    def reset_video_state(self, event=None):
        """Resets trim, keyframes, and crop state, keeping the cursor position."""
        if not self.video_path: return

        self.start_frame = 0
        self.end_frame = self.total_frames

        self.start_time = 0.0
        self.end_time = self.duration_seconds

        if self.video_path:
            self.update_all_ui()
            self.update_frame_preview(self.preview_frame_num)


    def trim_video_ffmpeg(self):
        original_extension = os.path.splitext(self.video_path)[1]
        save_path = self.default_filename_func if type(self.default_filename_func) is str else filedialog.asksaveasfilename(defaultextension=original_extension, initialdir=self.default_dir, initialfile=self.default_filename_func(os.path.splitext(os.path.basename(self.video_path))[0]), filetypes=[("Original Format", f"*{original_extension}"), ("All Files", "*.*")])
        if not save_path: return
        
        state = self.get_interpolated_state(self.start_time)
        if state['angle'] != 0:
            messagebox.showwarning("Warning", "Rotation is present but no keyframes are in the trimmed range. The output will not be rotated. Set at least two keyframes to render an animation.")

        orig_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        crop_w = int(state['w'] * orig_w)
        crop_h = int(state['h'] * orig_h)
        crop_x = int((state['cx'] - state['w']/2) * orig_w)
        crop_y = int((state['cy'] - state['h']/2) * orig_h)
        crop_filter = f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y}'
        command = (f'ffmpeg -i "{self.video_path}" '
                   f'-ss {self.start_time} -to {self.end_time} '
                   f'-vf "{crop_filter}" -c:v libx264 -c:a aac -y "{save_path}"')

        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            messagebox.showinfo("Success", f"Video successfully trimmed and saved to:\n{save_path}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Trimming Failed", f"FFmpeg returned an error:\n\n{e.stderr.decode()}")


    def trim_video_opencv(self):
        original_extension = os.path.splitext(self.video_path)[1]
        save_path = self.default_filename_func if type(self.default_filename_func) is str else filedialog.asksaveasfilename(defaultextension=original_extension, initialdir=self.default_dir, initialfile=self.default_filename_func(os.path.splitext(os.path.basename(self.video_path))[0]), filetypes=[("Original Format", f"*{original_extension}"), ("All Files", "*.*")])
        if not save_path: return

        if self.start_frame >= self.end_frame:
            messagebox.showwarning("Invalid Range", "The start frame must be before the end frame.")
            return


        render_cap = cv2.VideoCapture(self.video_path)
        if not render_cap.isOpened():
            messagebox.showerror("Error", "Could not open original video for rendering.")
            return

        source_w, source_h = int(render_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(render_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = render_cap.get(cv2.CAP_PROP_FPS)
        
        total_frames_to_render = self.end_frame - self.start_frame

        start_state = self.get_interpolated_state(self.start_frame / source_fps)
        output_w = int(start_state['w'] * source_w)
        output_h = int(start_state['h'] * source_h)
        if output_w % 2 != 0: output_w += 1
        if output_h % 2 != 0: output_h += 1

        progress_win = tk.Toplevel(self.master)
        progress_win.title("Rendering...")
        tk.Label(progress_win, text="Rendering video frames...").pack(padx=20, pady=10)
        render_bar = Progressbar(progress_win, orient=tk.HORIZONTAL, length=300, mode='determinate', maximum=total_frames_to_render)
        render_bar.pack(padx=20, pady=(0, 20))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, source_fps, (output_w, output_h))
        

        render_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        backlog = 0
        prev_frame = None
        for i in range(total_frames_to_render):
            ret, frame = render_cap.read()
            if not ret: break
            
            current_frame_num = self.start_frame + i
            current_time = current_frame_num / source_fps
            state = self.get_interpolated_state(current_time)
            zoomed_frame = self.render_zoomed_frame(frame, state, (source_w, source_h), (output_w, output_h))
            try:
                writer.write(zoomed_frame)
                if prev_frame is None:
                    if backlog >= 1:
                        for _ in range(backlog):
                            writer.write(zoomed_frame)
                prev_frame = zoomed_frame.copy() 
            except:
                if prev_frame is not None:
                    writer.write(prev_frame)
                else:
                    backlog += 1

            render_bar['value'] = i + 1
            progress_win.update()

        writer.release()
        render_cap.release()
        progress_win.destroy()
        

        mux_start_time = self.start_frame / source_fps
        mux_end_time = self.end_frame / source_fps

        mux_command = (f'ffmpeg -i "{TEMP_VIDEO_PATH}" '
                       f'-ss {mux_start_time} -to {mux_end_time} -i "{self.video_path}" '
                       f'-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -y "{save_path}"')
        try:
            subprocess.run(mux_command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            messagebox.showinfo("Success", f"Video successfully rendered and saved to:\n{save_path}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Muxing Failed", f"Could not combine video and audio. FFmpeg error:\n\n{e.stderr.decode()}")
        finally:
            if os.path.exists(TEMP_VIDEO_PATH): os.remove(TEMP_VIDEO_PATH)

    def render_zoomed_frame(self, frame, state, source_dims, target_dims):
        source_w, source_h = source_dims
        target_w, target_h = target_dims

        center_x_abs, center_y_abs = state['cx'] * source_w, state['cy'] * source_h
        crop_w_abs, crop_h_abs = int(state['w'] * source_w), int(state['h'] * source_h)

        M = cv2.getRotationMatrix2D((center_x_abs, center_y_abs), -state['angle'], 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (source_w, source_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        x1 = max(0, int(center_x_abs - crop_w_abs / 2))
        y1 = max(0, int(center_y_abs - crop_h_abs / 2))
        x2 = min(source_w, x1 + crop_w_abs)
        y2 = min(source_h, y1 + crop_h_abs)
        
        cropped_frame = rotated_frame[y1:y2, x1:x2]


        if (cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0) and self.prev_render_zoom_frame is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
            return self.prev_render_zoom_frame
        
        if cropped_frame.shape[1] != target_w or cropped_frame.shape[0] != target_h:
            self.prev_render_zoom_frame = cv2.resize(cropped_frame, (target_w, target_h), interpolation=cv2.INTER_AREA).copy()
            return cv2.resize(cropped_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        self.prev_render_zoom_frame = cropped_frame.copy()
        return cropped_frame



    def set_keyframe_event(self, event=None):
        if not self.video_path or self.is_playing: return
        key_time = round(self.preview_time, 3)
        self.keyframes[key_time] = self.crop_state.copy()
        # Add to history only if it's a new timestamp
        if key_time not in self.keyframe_history:
            self.keyframe_history.append(key_time)
        self.draw_timeline()

    def get_interpolated_state(self, time_seconds):
        if not self.keyframes:
            return self.crop_state.copy()
        
        sorted_times = sorted(self.keyframes.keys())
        
        if time_seconds <= sorted_times[0]: return self.keyframes[sorted_times[0]].copy()
        if time_seconds >= sorted_times[-1]: return self.keyframes[sorted_times[-1]].copy()

        for i in range(len(sorted_times) - 1):
            if sorted_times[i] <= time_seconds < sorted_times[i+1]:
                prev_time, next_time = sorted_times[i], sorted_times[i+1]
                break
        else: return self.crop_state.copy()
        
        prev_state, next_state = self.keyframes[prev_time], self.keyframes[next_time]
        duration = next_time - prev_time
        if duration == 0: return prev_state.copy()
        progress = (time_seconds - prev_time) / duration
        
        interpolated_state = {}
        for key in prev_state:
            if key == 'angle':
                p_angle, n_angle = prev_state[key], next_state[key]
                diff = n_angle - p_angle
                if diff > 180: p_angle += 360
                elif diff < -180: p_angle -= 360
                interpolated_state[key] = (p_angle + (n_angle - p_angle) * progress)
            else:
                interpolated_state[key] = prev_state[key] + (next_state[key] - prev_state[key]) * progress
        return interpolated_state

    def sync_crop_state_to_timeline(self):
        if not self.video_path: return
        self.crop_state = self.get_interpolated_state(self.preview_time)

    def draw_crop_overlay(self, base_image, state):
        w, h = self.image_display_size
        if w == 0 or h == 0: return base_image

        base_image = base_image.convert("RGBA")
        overlay_image = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_image)
        corners = self._get_rotated_corners(state, (w, h))
        
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).polygon(corners, fill=255)
        overlay_image.paste(Image.new("RGBA", (w, h), (0, 0, 0, 128)), (0, 0), Image.eval(mask, lambda p: 255 - p))

        draw.polygon(corners, outline="white", width=2)
        
        if self.crop_locked:
            rot_handle_start = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
            angle_rad = math.radians(state['angle'])
            rot_handle_end = (rot_handle_start[0] - 25 * math.sin(angle_rad), rot_handle_start[1] - 25 * math.cos(angle_rad))
            draw.line([rot_handle_start, rot_handle_end], fill="cyan", width=2)
            draw.ellipse([rot_handle_end[0]-6, rot_handle_end[1]-6, rot_handle_end[0]+6, rot_handle_end[1]+6], fill="cyan", outline="black")
        else:
            handle_size = 20
            for corner in corners:
                draw.rectangle([corner[0]-handle_size/2, corner[1]-handle_size/2, corner[0]+handle_size/2, corner[1]+handle_size/2], fill="white", outline="black")

        return Image.alpha_composite(base_image, overlay_image)

    def _get_rotated_corners(self, state, dims):
        w, h = dims
        center_x, center_y = state['cx'] * w, state['cy'] * h
        width, height = state['w'] * w, state['h'] * h
        angle_rad = math.radians(state['angle'])
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        half_w, half_h = width / 2, height / 2
        corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        return [(p[0] * cos_a - p[1] * sin_a + center_x, p[0] * sin_a + p[1] * cos_a + center_y) for p in corners_local]
    
    def _point_in_poly(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def on_crop_press(self, event):
        if not self.video_path or self.is_playing: return
        self.sync_crop_state_to_timeline()
        x, y = event.x, event.y

        self.crop_dragging_start_state = self.crop_state.copy()
        corners = self._get_rotated_corners(self.crop_state, self.image_display_size)
        handle_threshold = 15
        
        if not self.crop_locked:
            corner_names = ["tl", "tr", "br", "bl"]
            for i, corner in enumerate(corners):
                if math.hypot(x - corner[0], y - corner[1]) < handle_threshold:
                    self.crop_dragging_handle = corner_names[i]
                    self.drag_anchor = corners[(i + 2) % 4]
                    return
        else:
            rot_handle_start = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
            angle_rad = math.radians(self.crop_state['angle'])
            rot_handle_end = (rot_handle_start[0] - 25 * math.sin(angle_rad), rot_handle_start[1] - 25 * math.cos(angle_rad))
            if math.hypot(x - rot_handle_end[0], y - rot_handle_end[1]) < handle_threshold:
                self.crop_dragging_handle = "rotate"
                return

        if self.crop_locked and self._point_in_poly(x, y, corners):
            self.crop_dragging_handle = "move"
            self.drag_start_pos = (x, y)

    def on_crop_drag(self, event):
        if not self.crop_dragging_handle: return
        w, h = self.image_display_size
        if w == 0 or h == 0: return
        x, y = event.x, event.y
        start_state = self.crop_dragging_start_state
        
        if self.crop_dragging_handle == "move":
            dx = (x - self.drag_start_pos[0]) / w; dy = (y - self.drag_start_pos[1]) / h
            self.crop_state['cx'] = start_state['cx'] + dx; self.crop_state['cy'] = start_state['cy'] + dy
        elif self.crop_dragging_handle == "rotate":
            center_x, center_y = start_state['cx'] * w, start_state['cy'] * h
            angle = math.degrees(math.atan2(y - center_y, x - center_x)) + 90
            self.crop_state['angle'] = angle
        else: # Resizing corners
            angle_rad = math.radians(-start_state['angle'])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            new_cx = (x + self.drag_anchor[0]) / 2
            new_cy = (y + self.drag_anchor[1]) / 2
            anchor_local_x = (self.drag_anchor[0] - new_cx) * cos_a - (self.drag_anchor[1] - new_cy) * sin_a
            anchor_local_y = (self.drag_anchor[0] - new_cx) * sin_a + (self.drag_anchor[1] - new_cy) * cos_a
            new_w, new_h = abs(anchor_local_x * 2), abs(anchor_local_y * 2)

            if event.state & 0x0001:
                start_w_px = start_state['w'] * w
                start_h_px = start_state['h'] * h
                if start_h_px == 0: return
                aspect_ratio = start_w_px / start_h_px

                if new_h == 0: return
                #Constrains the new dimensions to the calculated aspect ratio
                if (new_w / new_h) > aspect_ratio:
                    new_w = new_h * aspect_ratio
                else:
                    new_h = new_w / aspect_ratio

            self.crop_state['cx'] = new_cx / w; self.crop_state['cy'] = new_cy / h
            self.crop_state['w'] = new_w / w; self.crop_state['h'] = new_h / h

        self._constrain_crop_state((w,h))
        self.update_frame_preview(self.preview_frame_num)

    def on_crop_release(self, event):
        if not self.crop_dragging_handle:
            return

        handle_type = self.crop_dragging_handle
        self.crop_dragging_handle = None
        self._constrain_crop_state(self.image_display_size)
        
        self.set_keyframe_event()

        is_resize = handle_type not in ["move", "rotate"]
        if is_resize and len(self.keyframes) > 1:
            new_ar = self.crop_state['w'] / self.crop_state['h'] if self.crop_state['h'] != 0 else 0
            sorted_times = sorted(self.keyframes.keys())
            prev_key_time = sorted_times[-2]
            ref_state = self.keyframes[prev_key_time]
            ref_ar = ref_state['w'] / ref_state['h'] if ref_state['h'] != 0 else 0
            
            if abs(new_ar - ref_ar) > 0.01:
                self.update_keyframes_aspect_ratio(new_ar)
        
        self.update_frame_preview(self.preview_frame_num)
        self.draw_timeline()
            
    def update_keyframes_aspect_ratio(self, new_aspect_ratio):
        if new_aspect_ratio == 0: return
        for t in self.keyframes:
            self.keyframes[t]['h'] = self.keyframes[t]['w'] / new_aspect_ratio
            
    def _constrain_crop_state(self, dims):
        w, h = dims
        if w == 0 or h == 0: return
        corners = self._get_rotated_corners(self.crop_state, dims)
        min_x, max_x = min(c[0] for c in corners), max(c[0] for c in corners)
        min_y, max_y = min(c[1] for c in corners), max(c[1] for c in corners)
        dx, dy = 0, 0
        if min_x < 0: dx = -min_x
        elif max_x > w: dx = w - max_x
        if min_y < 0: dy = -min_y
        elif max_y > h: dy = h - max_y
        if dx != 0 or dy != 0:
            self.crop_state['cx'] += dx / w
            self.crop_state['cy'] += dy / h
    
    def draw_timeline(self):
        self.timeline_canvas.delete("all")
        width, height = self.timeline_canvas.winfo_width(), self.timeline_canvas.winfo_height()
        if width < 2 or self.duration_seconds == 0: return
        self.timeline_canvas.create_rectangle(5, height * 0.4, width - 5, height * 0.6, fill="#cccccc", outline="")
        start_x, end_x = self.time_to_x(self.start_time), self.time_to_x(self.end_time)
        self.timeline_canvas.create_rectangle(start_x, height * 0.3, end_x, height * 0.7, fill="#99ccff", outline="")
        self.timeline_canvas.create_rectangle(start_x - 5, height * 0.2, start_x + 5, height * 0.8, fill="#007bff", outline="")
        self.timeline_canvas.create_rectangle(end_x - 5, height * 0.2, end_x + 5, height * 0.8, fill="#007bff", outline="")
        for t in self.keyframes:
            kx = self.time_to_x(t)
            self.timeline_canvas.create_polygon(kx, height*0.2, kx-4, height*0.4, kx, height*0.6, kx+4, height*0.4, fill="orange", outline="black")
        preview_x = self.time_to_x(self.preview_time)
        self.timeline_canvas.create_line(preview_x, 0, preview_x, height, fill="red", width=2)
        
    def full_reset(self):
        """Resets the entire animation state."""
        self.crop_state = {'cx': 0.5, 'cy': 0.5, 'w': 1.0, 'h': 1.0, 'angle': 0.0}
        self.keyframes = {}
        self.keyframe_history.clear()
        self.toggle_crop_lock_event(force_state=False)
        if self.video_path:
            self.update_all_ui()
            self.update_frame_preview(self.preview_frame_num)
    
    def reset_frame_to_default_event(self, event=None):
        """Resets the crop state for the current frame and sets a keyframe."""
        if not self.video_path: return
        self.crop_state = {'cx': 0.5, 'cy': 0.5, 'w': 1.0, 'h': 1.0, 'angle': 0.0}
        self.set_keyframe_event()
        self.update_frame_preview(self.preview_frame_num)

    def reset_frame_to_default_event2(self, event=None):
        """Resets the crop state for the current frame and sets a keyframe."""
        ow, oh = self.og_dimensions
        nw, nh = self.default_ar

        oar = ow/oh #ow wrt oh = 1
        nar = nw/nh #nw wrt nh = 1

        if nar < oar:
            w = nar/oar
            h = 1
        else:
            h = oar/nar
            w = 1

        if not self.video_path: return
        self.crop_state = {'cx': 0.5, 'cy': 0.5, 'w': w, 'h': h, 'angle': 0.0}
        self.set_keyframe_event()
        self.update_frame_preview(self.preview_frame_num)

    def undo_keyframe_event(self, event=None):
        """Removes the most recently placed keyframe."""
        if not self.keyframe_history:
            return
        last_key_time = self.keyframe_history.pop()
        if last_key_time in self.keyframes:
            del self.keyframes[last_key_time]
        
        self.sync_crop_state_to_timeline()
        self.update_all_ui()
        self.update_frame_preview(self.preview_frame_num)

    def save_screenshot_event(self, event=None):
        if self.video_path: self.save_screenshot()
        return "break"

    def save_screenshot(self, event=None):
        if not self.cap or not self.cap.isOpened() or self.is_playing: return
        self.stop_playback()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.preview_frame_num)

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read the frame to save screenshot.")
            return

        source_w, source_h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        current_state = self.get_interpolated_state(self.preview_time)
        rendered_frame = self.render_native_frame(frame, current_state, (source_w, source_h))


        if rendered_frame.size == 0:
            messagebox.showerror("Error", "Could not render frame, resulted in empty image.")
            return

        img_to_save = Image.fromarray(cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB))
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            initialdir=self.default_dir, 
            initialfile=self.default_pic_func(os.path.splitext(os.path.basename(self.video_path))[0]), 
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if not save_path: return

        try:
            img_to_save.save(save_path)
            messagebox.showinfo("Success", f"Screenshot saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"An error occurred while saving the image:\n\n{e}")


    def save_screenshot(self, event=None):
        if not self.cap or not self.cap.isOpened() or self.is_playing: return
        self.stop_playback()

        # open original highres Video specifically for this operation
        screenshot_cap = cv2.VideoCapture(self.video_path)
        if not screenshot_cap.isOpened():
             messagebox.showerror("Error", "Could not open original video source.")
             return

        screenshot_cap.set(cv2.CAP_PROP_POS_FRAMES, self.preview_frame_num)

        ret, frame = screenshot_cap.read()
        
        # Get dimensions from the high-res capture
        source_w = int(screenshot_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_h = int(screenshot_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        screenshot_cap.release()

        if not ret:
            messagebox.showerror("Error", "Could not read the frame to save screenshot.")
            return
        

        current_state = self.get_interpolated_state(self.preview_time)
        rendered_frame = self.render_native_frame(frame, current_state, (source_w, source_h))


        if rendered_frame.size == 0:
            messagebox.showerror("Error", "Could not render frame, resulted in empty image.")
            return

        img_to_save = Image.fromarray(cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB))
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            initialdir=self.default_dir, 
            initialfile=self.default_pic_func(os.path.splitext(os.path.basename(self.video_path))[0]), 
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if not save_path: return

        try:
            img_to_save.save(save_path)
            messagebox.showinfo("Success", f"Screenshot saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"An error occurred while saving the image:\n\n{e}")

    def toggle_crop_lock_event(self, event=None, force_state=None):
        if self.video_path:
            self.crop_locked = not self.crop_locked if force_state is None else force_state
            self.lock_button.config(text="Unlock Crop" if self.crop_locked else "Lock Crop", relief=tk.SUNKEN if self.crop_locked else tk.RAISED)
            self.update_frame_preview(self.preview_frame_num)
        return "break"
        
    def setup_key_bindings(self):
        self.master.bind('<space>', self.toggle_play_pause_event)
        self.master.bind('<Left>', lambda event: self.seek_frame(-1))
        self.master.bind('<Right>', lambda event: self.seek_frame(1))
        self.master.bind('a', self.set_start_trimmer_event)
        self.master.bind('s', self.set_end_trimmer_event)
        self.master.bind('r', self.reset_frame_to_default_event)
        self.master.bind('x', self.reset_frame_to_default_event2)
        self.master.bind('d', self.save_screenshot_event)
        self.master.bind('l', self.toggle_crop_lock_event)
        self.master.bind('k', self.set_keyframe_event)
        self.master.bind('z', self.undo_keyframe_event)
        self.master.bind('t', self.reset_video_state)

    def on_timeline_drag(self, event):
        if not self.timeline_dragging_handle or self.is_playing: return
        new_time = self.x_to_time(event.x)
        new_frame = int(new_time * self.fps)

        if self.timeline_dragging_handle == "start":
            self.start_frame = max(0, min(new_frame, self.end_frame))
            self.preview_frame_num = self.start_frame
        elif self.timeline_dragging_handle == "end":
            self.end_frame = min(self.total_frames, max(new_frame, self.start_frame))
            self.preview_frame_num = self.end_frame
        elif self.timeline_dragging_handle == "preview":
            self.preview_frame_num = max(0, min(new_frame, self.total_frames))

        #Updates time variables for UI display
        self.start_time = self.start_frame / self.fps
        self.end_time = self.end_frame / self.fps
        self.preview_time = self.preview_frame_num / self.fps

        self.stop_playback()
        self.update_all_ui()
        self.update_frame_preview(self.preview_frame_num)

    def on_timeline_release(self, event):
        if self.timeline_dragging_handle:
            self.timeline_dragging_handle = None
            self.sync_crop_state_to_timeline()

    def toggle_play_pause(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.is_playing = True
            self.play_pause_button.config(text="❚❚ Pause")

            if self.is_paused:
                self.audio_player.unpause()
                self.is_paused = False
            else:
                if self.preview_frame_num >= self.end_frame or self.preview_frame_num < self.start_frame:
                    self.preview_frame_num = self.start_frame
                

                self.preview_time = self.preview_frame_num / self.fps if self.fps > 0 else 0
                self.play_audio_segment()

            # Store the state at the moment playback begins
            self.playback_start_system_time = time.time()
            self.playback_start_video_time = self.preview_time
            self.video_update_loop()

    def stop_playback(self):
        was_playing = self.is_playing
        self.is_playing, self.is_paused = False, False
        self.play_pause_button.config(text="▶ Play")
        self.audio_player.stop()
        if was_playing:
            self.sync_crop_state_to_timeline()

    def video_update_loop(self):
        if not self.is_playing: return

        elapsed_system_time = time.time() - self.playback_start_system_time
        
        # Get theoretical time
        self.preview_time = self.playback_start_video_time + elapsed_system_time
        

        self.preview_frame_num = int(self.preview_time * self.fps)

        if self.preview_frame_num >= self.end_frame or not self.audio_player.is_busy():
            # Clamp the final position to the end_frame to avoid overshooting
            self.preview_frame_num = min(self.preview_frame_num, self.end_frame)
            self.preview_time = self.preview_frame_num / self.fps if self.fps > 0 else 0
            self.stop_playback()
        
        self.update_all_ui()
        

        self.update_frame_preview(self.preview_frame_num)
        
        if self.is_playing: 
            self.after(25, self.video_update_loop)

    def play_audio_segment(self):
        self.audio_player.stop()
        extract_cmd = (f'ffmpeg -ss {self.preview_time} -to {self.end_time} -i "{self.video_path}" -vn -y -acodec pcm_s16le -ar 44100 -ac 2 "{TEMP_AUDIO_PATH}"')
        try:
            subprocess.run(extract_cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.audio_player.play(TEMP_AUDIO_PATH)
        except (subprocess.CalledProcessError, FileNotFoundError): messagebox.showwarning("Audio Error", "Could not extract audio for playback.")
    
    def release_resources(self): 
        self.stop_playback()
        self.audio_player.close()
        self.master.destroy()
        cleanup()

    def on_timeline_press(self, event):
        if self.is_playing: 
            return
        
        x, start_x, end_x, preview_x = event.x, self.time_to_x(self.start_time), self.time_to_x(self.end_time), self.time_to_x(self.preview_time)
        
        if abs(x - preview_x) < 12: 
            self.timeline_dragging_handle = "preview"
        elif abs(x - start_x) < 8: 
            self.timeline_dragging_handle = "start"
        elif abs(x - end_x) < 8: 
            self.timeline_dragging_handle = "end"
        else: 
            self.timeline_dragging_handle = "preview"; self.on_timeline_drag(event)
    
    def update_all_ui(self): 
        self.update_time_labels(); self.draw_timeline()
    
    def update_time_labels(self):
        self.start_time_label.config(text=f"Start: {self.format_time(self.start_time)}"); self.end_time_label.config(text=f"End: {self.format_time(self.end_time)}"); self.preview_time_label.config(text=f"Cursor: {self.format_time(self.preview_time)}")
    

    def format_time(self, seconds): 
        millis = int((seconds - int(seconds)) * 100)
        seconds = int(seconds)
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:02}"
    
    def time_to_x(self, time_val): 
        width = self.timeline_canvas.winfo_width() - 10
        return 5 + (min(self.duration_seconds, max(0, time_val)) / self.duration_seconds) * width if self.duration_seconds > 0 else 5
    
    def x_to_time(self, x_val): 
        width = self.timeline_canvas.winfo_width() - 10; return (max(0, x_val - 5) / width) * self.duration_seconds if width > 0 else 0
    
    def toggle_play_pause_event(self, event=None):
        if self.video_path: 
            self.toggle_play_pause()
        return "break"
    
    def seek_frame(self, direction):
        if not self.video_path or self.fps == 0 or self.is_playing: 
            return
        

        new_frame_num = self.preview_frame_num + direction
        self.preview_frame_num = max(0, min(new_frame_num, self.total_frames - 1))
        

        self.preview_time = self.preview_frame_num / self.fps

        self.update_all_ui()
        self.update_frame_preview(self.preview_frame_num)
        self.sync_crop_state_to_timeline()

    def set_start_trimmer_event(self, event=None):
        if not self.video_path or self.is_playing: 
            return
        self.start_frame = self.preview_frame_num
        if self.start_frame > self.end_frame:
            self.end_frame = self.start_frame
        
        self.start_time = self.start_frame / self.fps
        self.end_time = self.end_frame / self.fps

        self.stop_playback()
        self.update_all_ui()

    def set_end_trimmer_event(self, event=None):
        if not self.video_path or self.is_playing: 
            return
        self.end_frame = self.preview_frame_num
        if self.end_frame < self.start_frame:
            self.start_frame = self.end_frame


        self.start_time = self.start_frame / self.fps
        self.end_time = self.end_frame / self.fps

        self.stop_playback()
        self.update_all_ui()


docstring = """
--------------------------------------------------
          Scene Builder - Usage Guide
--------------------------------------------------

This tool allows you to trim, crop, and create animated
camera movements (pan, zoom, rotate) for your videos.

--- Workflow ---

1.  Load a video using the "Load Video" button.
2.  Use the mouse to drag the blue start/end handles on the
    timeline, or move the red cursor and press 'A' (start)
    and 'S' (end) to define the trim section.
3.  Unlock the crop box ('L' key) to resize it. Hold 'Shift'
    while dragging a corner to lock the aspect ratio. This
    aspect ratio will become the project's aspect ratio.
4.  Lock the crop box ('L' key). Now you can move it around
    (pan) or rotate it.
5.  Position the crop box and press 'K' to create a keyframe.
6.  Move the timeline cursor to another point in time.
7.  Reposition/resize/rotate the crop box and press 'K' again.
    An animation will now be created between these two points.
8.  Click "Trim and Save Video" to render the final clip.

--- Keybindings ---

[ Playback & Timeline ]
  Spacebar      : Toggle Play/Pause.
  Left Arrow    : Seek one frame backward.
  Right Arrow   : Seek one frame forward.
  Mouse Drag    : Scrub the red timeline cursor.

[ Trimming ]
  A             : Set the Start Time to the cursor's position.
  S             : Set the End Time to the cursor's position.

[ Cropping & Animation ]
  L             : Toggle between UNLOCKED and LOCKED crop mode.

  -- UNLOCKED Mode (for Resizing) --
  Drag Corner   : Resize the crop box freely.
  Shift+Drag    : Resize while maintaining the aspect ratio.
                  (This sets the aspect ratio for the whole animation).

  -- LOCKED Mode (for Pan & Rotate) --
  Drag Body     : Pan the crop box around the frame.
  Drag Handle   : Rotate the crop box.

[ Keyframing ]
  K             : Set a keyframe at the current time with the
                  current crop settings.
  Z             : Undo the previously set keyframe.
  R             : Reset the crop for the current frame to the
                  default (full screen, no rotation) and set a
                  keyframe there.
  X             : Set Aspect Ratio to the final video's aspect ratio.
  T             : Resets video for next animation.

[ Saving ]
  D             : Save a high-resolution screenshot of the
                  currently displayed (cropped) frame.
--------------------------------------------------
"""


def launch(file_func = lambda x:f"{x}_trimmed.mp4", default_dir:str = None):
    make_dpi_aware()
    root = tk.Tk()
    app = SceneSplitter(master=root, default_filename_func=file_func, default_dir=default_dir, default_aspect_ratio=settings.VIDEO_SIZE)
    root.protocol("WM_DELETE_WINDOW", app.release_resources)
    app.mainloop()

def launch_adv(file_func = lambda x:f"{x}_trimmed.mp4", default_dir:str = None, default_pic = lambda x: f"{x}_screenshot.png"):
    make_dpi_aware()
    print(docstring)
    root = tk.Tk()
    app = SceneSplitter(master=root, default_filename_func=file_func, default_dir=default_dir, default_pic=default_pic, default_aspect_ratio=settings.VIDEO_SIZE)
    root.protocol("WM_DELETE_WINDOW", app.release_resources)
    app.mainloop()

if __name__ == '__main__':
    launch_adv()