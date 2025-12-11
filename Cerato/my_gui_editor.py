import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
import shutil

class MediaHelperApp:
    def __init__(self, root, input_paths, output_paths, pre_call, image_editors, video_editors):
        self.root = root
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.pre_call = pre_call
        self.image_editors = image_editors
        self.video_editors = video_editors

        self.current_file_index = 0
        self.hot_reload_timer = None
        self.original_image = None
        self.current_sliders = {}
        self.slider_value_labels = {}
        self.current_editors = []


        self.is_hot_reloading = False
        self.is_video_seeking = False
        self.is_filter_dragging = False

        self.last_requested_video_state = None
        

        self.current_temp_video_path = None
        self.video_capture = None
        self.video_paused = True
        self.was_playing_before_seek = False
        self.playback_after_id = None
        self.video_delay_ms = 40

        self.setup_ui()
        self.root.after(100, self.load_media)

    def setup_ui(self):
        self.root.title("Media Helper")
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.on_closing())

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.media_frame = ttk.Frame(main_frame)
        self.media_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.media_label = ttk.Label(self.media_frame)
        self.media_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.video_controls_frame = ttk.Frame(self.media_frame)
        self.play_pause_button = ttk.Button(self.video_controls_frame, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.seek_var = tk.DoubleVar()
        self.seek_bar = ttk.Scale(self.video_controls_frame, from_=0, variable=self.seek_var, orient=tk.HORIZONTAL)
        self.seek_bar.bind("<ButtonPress-1>", self.on_video_seek_start)
        self.seek_bar.bind("<ButtonRelease-1>", self.on_video_seek_end)
        self.seek_bar.bind("<B1-Motion>", self.on_video_seek_drag)
        self.seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=5)

        sidebar = ttk.Frame(main_frame, width=350)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        sidebar.pack_propagate(False)

        header_frame = ttk.Frame(sidebar)
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.editor_variable = tk.StringVar()
        self.editor_dropdown = ttk.Combobox(header_frame, textvariable=self.editor_variable, state="readonly")
        self.editor_dropdown.pack(fill=tk.X, expand=True, side=tk.TOP)
        self.editor_dropdown.bind("<<ComboboxSelected>>", self.on_editor_selected)
        self.status_label = ttk.Label(header_frame, text="Ready", font=("Segoe UI", 9))
        self.status_label.pack(side=tk.TOP, fill=tk.X, pady=(5,0))

        button_container = ttk.Frame(sidebar)
        button_container.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        nav_button_frame = ttk.Frame(button_container)
        nav_button_frame.pack(fill=tk.X, expand=True)
        self.prev_button = ttk.Button(nav_button_frame, text="Prev", command=self.prev_media)
        self.prev_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(nav_button_frame, text="Reset", command=self.reset_sliders).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.next_button = ttk.Button(nav_button_frame, text="Next", command=self.next_media)
        self.next_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        save_button_frame = ttk.Frame(button_container)
        save_button_frame.pack(fill=tk.X, expand=True, pady=(5,0))
        self.save_prev_button = ttk.Button(save_button_frame, text="Save & Prev", command=lambda: self.save_media('prev'))
        self.save_prev_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(save_button_frame, text="Save", command=lambda: self.save_media()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.save_next_button = ttk.Button(save_button_frame, text="Save & Next", command=lambda: self.save_media('next'))
        self.save_next_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        quit_button_frame = ttk.Frame(button_container)
        quit_button_frame.pack(fill=tk.X, expand=True, pady=(10,0))
        ttk.Button(quit_button_frame, text="Quit Application", command=self.on_closing).pack(expand=True, fill=tk.X)
        
        slider_container = ttk.Frame(sidebar)
        slider_container.pack(fill=tk.BOTH, expand=True, pady=5)
        self.slider_canvas = tk.Canvas(slider_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(slider_container, orient="vertical", command=self.slider_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.slider_canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.slider_canvas.configure(scrollregion=self.slider_canvas.bbox("all")))
        self.slider_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.slider_canvas.configure(yscrollcommand=scrollbar.set)
        self.slider_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_status(self, text, color):
        self.root.after(0, lambda: self.status_label.config(text=text, foreground=color))

    def _stop_current_video(self):
        if self.playback_after_id: self.root.after_cancel(self.playback_after_id)
        self.playback_after_id = None
        if self.video_capture and self.video_capture.isOpened(): self.video_capture.release()
        self.video_capture = None
        self.video_paused = True
        self.current_temp_video_path = None

    def load_media(self):
        self._stop_current_video()
        self._update_button_states()
        if not (0 <= self.current_file_index < len(self.input_paths)): self.on_closing(); return
        filepath = self.input_paths[self.current_file_index]
        file_extension = os.path.splitext(filepath)[1].lower()
        if file_extension in ['.png', '.jpg', '.jpeg']:
            self.video_controls_frame.place_forget()
            self.load_image(filepath)
        elif file_extension == '.mp4':
            self.video_controls_frame.place(relx=0.0, rely=1.0, relwidth=1.0, anchor=tk.SW)
            self.load_video(filepath)

    def load_image(self, filepath):
        self.original_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        self.update_media_display(self.original_image)
        self.setup_editors(self.image_editors)

    def load_video(self, filepath):
        self.setup_editors(self.video_editors)
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        cap.release()
        if ret: self.update_media_display(frame)
        self.apply_edits()

    def setup_editors(self, editors):
        self.current_editors = editors
        editor_names = [e[0] for e in editors]
        self.editor_dropdown['values'] = editor_names
        if editor_names: self.editor_dropdown.current(0)
        self.on_editor_selected()

    def on_editor_selected(self, event=None):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.current_sliders.clear(); self.slider_value_labels.clear()
        editor_name = self.editor_variable.get()
        editor_config = next((e[2] for e in self.current_editors if e[0] == editor_name), None)
        if editor_config:
            for option, (start, end, default) in editor_config.items():
                row = ttk.Frame(self.scrollable_frame)
                row.pack(fill=tk.X, expand=True, pady=5, padx=5)
                label_frame = ttk.Frame(row); label_frame.pack(fill=tk.X)
                ttk.Label(label_frame, text=option).pack(side=tk.LEFT)
                value_label = ttk.Label(label_frame, text=f"{default:.2f}"); value_label.pack(side=tk.RIGHT)
                self.slider_value_labels[option] = value_label
                slider_var = tk.DoubleVar(value=default)
                slider = ttk.Scale(row, from_=start, to=end, variable=slider_var, orient=tk.HORIZONTAL)
                slider.bind("<ButtonPress-1>", lambda e, s=slider, v=slider_var, o=option: self._on_filter_slider_press(e, s, v, o))
                slider.bind("<ButtonRelease-1>", self._on_filter_slider_release)
                slider.bind("<B1-Motion>", lambda e, s=slider, v=slider_var, o=option: self._on_filter_slider_drag(e, s, v, o))
                slider.pack(fill=tk.X, expand=True)
                self.current_sliders[option] = slider_var
        self.apply_edits()

    def on_slider_change(self, value, option_name):
        self.slider_value_labels[option_name].config(text=f"{float(value):.2f}")
        if self.hot_reload_timer: 
            self.hot_reload_timer.cancel()
        
        filepath = self.input_paths[self.current_file_index]
        file_ext = os.path.splitext(filepath)[1].lower()

        self.hot_reload_timer = threading.Timer(1.0 if file_ext == '.mp4' else 0.5, self.apply_edits)
        self.hot_reload_timer.start()

    def apply_edits(self):
        
        if self.is_hot_reloading: 
            print("hotreloading...")
            return
        print("Applying reload")
        editor_name = self.editor_variable.get()
        editor_func = next((e[1] for e in self.current_editors if e[0] == editor_name), None)
        
        if not editor_func: 
            self._set_status("Editor not found", "red")
            return
        
        current_state = {opt: var.get() for opt, var in self.current_sliders.items()}
        
        filepath = self.input_paths[self.current_file_index]
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            self._set_status("Loading...", "orange")
            if self.original_image is not None:
                new_image = editor_func(self.original_image.copy(), **current_state)
                self.update_media_display(new_image)
                if {opt: var.get() for opt, var in self.current_sliders.items()} != current_state:
                    self.apply_edits()
        elif file_ext == '.mp4':
            self.is_hot_reloading = True
            self._set_status("Loading...", "orange")
            self._stop_current_video()
            self._update_label_image(None) 
            current_state['hot_reload'] = True
            threading.Thread(target=self._hot_reload_worker, args=(editor_func, filepath, current_state), daemon=True).start()
            self.last_requested_video_state = current_state

    def _hot_reload_worker(self, editor_func, filepath, kwargs):
        try:
            new_temp_path = editor_func(filepath, **kwargs)
            self.root.after(0, self._start_video_preview, new_temp_path)
        except Exception as e:
            print(f"Error in hot reload worker: {e}"); self._set_status("Reload Error", "red")
        finally:
            self.is_hot_reloading = False

    def update_media_display(self, cv2_image):
        if cv2_image is None: self._update_label_image(None); return
        frame_width = self.media_frame.winfo_width()
        frame_height = self.media_frame.winfo_height()
        if frame_width <= 1 or frame_height <= 1: return
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 4:
            img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        

        source_width, source_height = img_pil.size
        box_width = frame_width - 20
        box_height = frame_height - 60

        if box_width <= 0 or box_height <= 0: return

        source_aspect = source_width / source_height
        box_aspect = box_width / box_height

        if source_aspect > box_aspect:
            new_width = box_width
            new_height = int(new_width / source_aspect)
        else:
            new_height = box_height
            new_width = int(new_height * source_aspect)
        
        resized_img = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=resized_img)

        
        self.root.after(0, self._update_label_image, photo)
        if not self.is_hot_reloading: self._set_status("Loaded", "green")

    def _update_label_image(self, photo):
        self.media_label.config(image=photo)
        self.media_label.image = photo

    def _start_video_preview(self, new_temp_path):
        current_gui_state = {opt: var.get() for opt, var in self.current_sliders.items()}
        if current_gui_state != self.last_requested_video_state:
            current_gui_state["hot_reload"] = True
        if current_gui_state != self.last_requested_video_state:
            current_gui_state["hot_reload"] = False
        if current_gui_state != self.last_requested_video_state:       
            # The user has changed sliders while we were making the preview. Discard this old preview and trigger a new reload.
            print(f"User has change during preview {self.last_requested_video_state} -> {current_gui_state}")
            self.apply_edits()
            return

        
        self._stop_current_video()
        if not os.path.exists(new_temp_path):
            self._set_status(f"Preview file not found", "red"); return
        
        self.current_temp_video_path = new_temp_path
        self.video_capture = cv2.VideoCapture(new_temp_path)
        
        if not self.video_capture.isOpened():
            self._set_status("Failed to open video", "red"); return
        
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_delay_ms = int(1000 / fps) if fps > 0 else 40
        self.seek_bar.config(to=total_frames - 1 if total_frames > 0 else 0)
        self.seek_var.set(0)
        self.video_paused = True
        self.play_pause_button.config(text="Play")
        self._show_seek_frame(0)

    def toggle_play_pause(self):
        if self.video_paused:
            self.video_paused = False; self.play_pause_button.config(text="Pause")
            self._video_playback_loop()
        else:
            self.video_paused = True; self.play_pause_button.config(text="Play")
            if self.playback_after_id: self.root.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def _video_playback_loop(self):
        if self.video_paused or not self.video_capture: return
        ret, frame = self.video_capture.read()
        if not ret: self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = self.video_capture.read()
        if not ret: self._stop_current_video(); return
        self.update_media_display(frame)
        if not self.is_video_seeking:
            current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            self.seek_var.set(current_frame)
        self.playback_after_id = self.root.after(self.video_delay_ms, self._video_playback_loop)

    def _jump_to_click(self, slider, var, mouse_x):
        from_ = slider.cget("from")
        to = slider.cget("to")
        widget_width = slider.winfo_width()
        if widget_width > 0:
            new_value = from_ + (to - from_) * (mouse_x / widget_width)
            new_value = max(from_, min(new_value, to))
            var.set(new_value)
            return new_value
        return var.get()

    def _on_filter_slider_press(self, event, slider, var, option_name):
        self.is_filter_dragging = True
        new_value = self._jump_to_click(slider, var, event.x)
        self.on_slider_change(new_value, option_name)
        return "break"

    def _on_filter_slider_drag(self, event, slider, var, option_name):
        if self.is_filter_dragging:
            new_value = self._jump_to_click(slider, var, event.x)
            self.on_slider_change(new_value, option_name)

    def _on_filter_slider_release(self, event):
        self.is_filter_dragging = False

    def on_video_seek_start(self, event):
        self.is_video_seeking = True
        self.was_playing_before_seek = not self.video_paused
        if self.was_playing_before_seek: self.toggle_play_pause()
        new_value = self._jump_to_click(self.seek_bar, self.seek_var, event.x)
        self._show_seek_frame(int(new_value))
        return "break"

    def on_video_seek_end(self, event):
        if not self.is_video_seeking: return
        self.is_video_seeking = False
        self._show_seek_frame(int(self.seek_var.get()))
        if self.was_playing_before_seek: self.toggle_play_pause()

    def on_video_seek_drag(self, event):
        if self.is_video_seeking:
            new_value = self._jump_to_click(self.seek_bar, self.seek_var, event.x)
            self._show_seek_frame(int(new_value))

    def _show_seek_frame(self, frame_number):
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video_capture.read()
            if ret: self.update_media_display(frame)
    
    def _update_button_states(self):
        is_first = self.current_file_index == 0
        is_last = self.current_file_index >= len(self.input_paths) - 1
        
        self.prev_button.config(state='disabled' if is_first else 'normal')
        self.save_prev_button.config(state='disabled' if is_first else 'normal')
        self.next_button.config(state='disabled' if is_last else 'normal')
        self.save_next_button.config(state='disabled' if is_last else 'normal')

    def next_media(self):
        if self.current_file_index < len(self.input_paths) - 1:
            self.current_file_index += 1; self.load_media()

    def prev_media(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1; self.load_media()
    
    def reset_sliders(self):
        editor_name = self.editor_variable.get()
        editor_config = next((e[2] for e in self.current_editors if e[0] == editor_name), None)
        if editor_config:
            for option, (start, end, default) in editor_config.items():
                if option in self.current_sliders:
                    self.current_sliders[option].set(default)
                    self.on_slider_change(default, option)
            if self.hot_reload_timer: self.hot_reload_timer.cancel()
            self.hot_reload_timer = threading.Timer(0.1, self.apply_edits)
            self.hot_reload_timer.start()

    def save_media(self, move_direction=None):
        self._stop_current_video()
        input_path = self.input_paths[self.current_file_index]
        output_path = self.output_paths[self.current_file_index]
        self.pre_call(output_path)
        editor_name = self.editor_variable.get()
        editor_func = next((e[1] for e in self.current_editors if e[0] == editor_name), None)
        if not editor_func: return
        kwargs = {opt: var.get() for opt, var in self.current_sliders.items()}
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.png', '.jpg', '.jpeg']:
            if self.original_image is not None:
                final_image = editor_func(self.original_image.copy(), **kwargs)
                cv2.imwrite(output_path, final_image)
        elif file_ext == '.mp4':
            self._set_status("Saving Video...", "blue")
            kwargs['hot_reload'] = False
            temp_video_path = editor_func(input_path, **kwargs)
            if os.path.exists(output_path): os.remove(output_path)
            shutil.move(temp_video_path, output_path)
            self._set_status("Save Complete", "green")
        if move_direction == 'next': self.next_media()
        elif move_direction == 'prev': self.prev_media()
        
    def on_closing(self):
        self._stop_current_video()
        if self.hot_reload_timer: self.hot_reload_timer.cancel()
        self.root.destroy()

def start_helper_app(input_paths, output_paths, pre_call, image_editors, video_editors):
    if not input_paths: print("No input files provided."); return
    root = tk.Tk()
    app = MediaHelperApp(root, input_paths, output_paths, pre_call, image_editors, video_editors)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()



# -- Main script --

import ffxcc
import transformers
import cv2
import progress_func as pf


temporary_file_location = "temp.mp4"
def video_editor_builder(filter_effect = ffxcc.cc_main, progress_func = lambda x: x, zoom_prog_func = lambda x: x, provide_last_second = False):
    """Takes in function that can produce output frames, yielding them with input cap and kwargs-> output BGR frame. This will add in cc automatically"""
    def new_func(input_path, hot_reload=False, max_zoom = 1, speed = 1, **kwargs):
        global temporary_file_location
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("Cannot open input video")

        if hot_reload:
            print("Enabled hot reload | Currently no implementation though...")
        print(f"Kwargs: {kwargs}")
        

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temporary_file_location, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_to_allow = {int(speed * i) for i in range(total_frames)}

        if provide_last_second:
            last_second = sorted(list(frames_to_allow))
            if len(last_second) >= fps:
                last_second = last_second[int(len(last_second) - fps): ]
            else:
                last_second = [-1 for i in range(int(fps - len(last_second)))] + last_second

        for frame_num in range(total_frames): #idk buffer ig
            if frame_num not in frames_to_allow:
                continue

            if provide_last_second:
                try:
                    prog = last_second.index(frame_num)/(len(last_second) - 1)
                except ValueError:
                    prog = -1

            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, max(0, int(total_frames * progress_func(frame_num/total_frames)))))
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_zoom != 1:
                frame = transformers.zoom_transform(frame, zoom_prog_func(frame_num/total_frames), max_zoom)
            
            if provide_last_second:
                new_frame = filter_effect(frame, progress = prog,**kwargs)
            else:
                new_frame = filter_effect(frame, **kwargs)
            out.write(new_frame)

        cap.release()
        out.release()

        return temporary_file_location

    return new_func


image_editors = [
    ("main cc", ffxcc.cc_main, {"strength": (-3, 5, 1)}),
    ("cc 1", ffxcc.cc_1, {"vignette_strength": (0, 0.6, 0.04), "highlights": (-0.2, 0.1, -0.05), "vibrance": (-2, 4, 0.74)}),
]

video_editors = [
    #Simple filters
    ("no cc", video_editor_builder(lambda x: x), {'speed':(1,2.5, 1)}),
    ("simple_cc", video_editor_builder(ffxcc.cc_main, provide_last_second=True), {'strength': image_editors[0][2]['strength'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),
    ("complex_cc", video_editor_builder(ffxcc.cc_1, provide_last_second=True), {"vignette_strength": image_editors[1][2]['vignette_strength'], "highlights": image_editors[1][2]['highlights'], "vibrance": image_editors[1][2]['vibrance'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),

    #Back and forth
    ("b&f + no cc", video_editor_builder(lambda x: x, pf.parabolic_tepee), {'speed':(1,2.5, 1)}),
    ("b&f + simple_cc", video_editor_builder(ffxcc.cc_main, pf.parabolic_tepee, provide_last_second=True), {'strength': image_editors[0][2]['strength'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),
    ("b&f + complex_cc", video_editor_builder(ffxcc.cc_1, pf.parabolic_tepee, provide_last_second=True), {"vignette_strength": image_editors[1][2]['vignette_strength'], "highlights": image_editors[1][2]['highlights'], "vibrance": image_editors[1][2]['vibrance'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),

    #Back and forth variation
    ("b&f2 + no cc", video_editor_builder(lambda x: x, pf.tepee), {'speed':(1,2.5, 1)}),
    ("b&f2 + simple_cc", video_editor_builder(ffxcc.cc_main, pf.tepee, provide_last_second=True), {'strength': image_editors[0][2]['strength'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),
    ("b&f2 + complex_cc", video_editor_builder(ffxcc.cc_1, pf.tepee, provide_last_second=True), {"vignette_strength": image_editors[1][2]['vignette_strength'], "highlights": image_editors[1][2]['highlights'], "vibrance": image_editors[1][2]['vibrance'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),

    #Pan in
    ("pan_in + no cc", video_editor_builder(lambda x: x, zoom_prog_func=lambda x:x), {'max_zoom':(1,3, 1.31), 'speed':(1,2.5, 1)}),
    ("pan_in + simple_cc", video_editor_builder(ffxcc.cc_main, zoom_prog_func=lambda x:x, provide_last_second=True), {'max_zoom':(1,3, 1.31), 'strength': image_editors[0][2]['strength'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),
    ("pan_in + complex_cc", video_editor_builder(ffxcc.cc_1, zoom_prog_func=lambda x:x, provide_last_second=True), {'max_zoom':(1,3, 1.31), "vignette_strength": image_editors[1][2]['vignette_strength'], "highlights": image_editors[1][2]['highlights'], "vibrance": image_editors[1][2]['vibrance'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),

    #Exp Pan in
    ("exp_pan_in + no cc", video_editor_builder(lambda x: x, zoom_prog_func=lambda x:x**2), {'max_zoom':(1,3, 1.31), 'speed':(1,2.5, 1)}),
    ("exp_pan_in + simple_cc", video_editor_builder(ffxcc.cc_main, zoom_prog_func=lambda x:x**2, provide_last_second=True), {'max_zoom':(1,3, 1.31), 'strength': image_editors[0][2]['strength'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),
    ("exp_pan_in + complex_cc", video_editor_builder(ffxcc.cc_1, zoom_prog_func=lambda x:x**2, provide_last_second=True), {'max_zoom':(1,3, 1.31), "vignette_strength": image_editors[1][2]['vignette_strength'], "highlights": image_editors[1][2]['highlights'], "vibrance": image_editors[1][2]['vibrance'], 'speed':(1,2.5, 1), 'contour_mode':(0.0,2.0,0.0)}),

    #Shock zoom
    #("shock_zoom + simple_cc", video_editor_builder(ffxcc.cc_main, zoom_prog_func=lambda x:x**10), {'max_zoom':(1,20, 5), 'strength': image_editors[0][2]['strength']}),
    #("shock_zoom + complex_cc", video_editor_builder(ffxcc.cc_1, zoom_prog_func=lambda x:x**10), {'max_zoom':(1,20, 5), "vignette_strength": image_editors[1][2]['vignette_strength'], "highlights": image_editors[1][2]['highlights'], "vibrance": image_editors[1][2]['vibrance']}),
]

def launch(input_paths, output_paths, pre_call):
    global temporary_file_location

    print('Starting mge with: ')
    print(temporary_file_location)
    print(input_paths)
    print(output_paths)


    start_helper_app(input_paths, output_paths, pre_call, image_editors, video_editors)


if __name__ == "__main__":
    input_paths = ["input.mp4", "car_portrait.png", "bread.png", "car_rb.png", "input_1.mp4"]
    output_paths = ["input_out.mp4", "car_portrait_out.png", "bread_out.png", "car_rb_out.png", "input_1_out.mp4"]
    pre_call = lambda x:print(f"Saving {x} : {input('>>')}")

    launch(input_paths, output_paths, pre_call)