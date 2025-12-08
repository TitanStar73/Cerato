import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

class PerlinNoise:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        p = np.arange(256, dtype=int)
        rng.shuffle(p)
        self.p = np.stack([p, p]).flatten()
        self.g = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]], dtype=float)

    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t):
        return a + t * (b - a)

    def _grad(self, hash_val, x, y):
        g_vec = self.g[hash_val % 8]
        return g_vec[0] * x + g_vec[1] * y

    def noise(self, x, y):
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)
        p = self.p
        aa = p[X] + Y
        ab = p[X] + Y + 1
        ba = p[X + 1] + Y
        bb = p[X + 1] + Y + 1
        n00 = self._grad(p[aa], xf, yf)
        n01 = self._grad(p[ab], xf, yf - 1)
        n10 = self._grad(p[ba], xf - 1, yf)
        n11 = self._grad(p[bb], xf - 1, yf - 1)
        x1 = self._lerp(n00, n10, u)
        x2 = self._lerp(n01, n11, u)
        return self._lerp(x1, x2, v)



def get_smart_color_from_image(image_path, saturation_threshold=0.2):
    try:
        img_cv = cv2.imread(image_path)
        pixels = img_cv.reshape(-1, 3)
        hsv_pixels = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2HSV)[0]
        avg_saturation = np.mean(hsv_pixels[:, 1]) / 255.0
        if avg_saturation < saturation_threshold: return (255, 255, 255)
        pixels_float = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, center = cv2.kmeans(pixels_float, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_bgr = np.uint8(center[0])
        dominant_hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        neon_hsv = np.uint8([dominant_hsv[0], 255, 255])
        neon_bgr = cv2.cvtColor(np.uint8([[neon_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        return (int(neon_bgr[2]), int(neon_bgr[1]), int(neon_bgr[0]))
    except Exception: return (255, 255, 255)

def ease_out_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(x - 1, 3) + c1 * pow(x - 1, 2)

def create_feature_glow_frame(
    frame_dims, 
    base_contours,
    noise_field_x,
    noise_field_y,
    bg_color=(0, 0, 0),
    wisp=1.0,
    solid_color=None,
    phase_shift=0.0,
    glow_source_thickness_ratio=0.002,
    core_thickness_ratio=0.0002,
    glow_radius_ratio=0.005,
):
    h, w = frame_dims
    diagonal = np.sqrt(h**2 + w**2)
    motion_wisp = ease_out_back(wisp)
    opacity_wisp = 1 #wisp
    current_glow_radius = max(1, int(diagonal * glow_radius_ratio))
    current_source_thickness = max(1, int(diagonal * glow_source_thickness_ratio))
    current_core_thickness = max(1, int(diagonal * core_thickness_ratio))
    
    glow_source_pil = create_elastic_wireframe(h, w, base_contours, noise_field_x, noise_field_y, current_source_thickness, solid_color, phase_shift, motion_wisp)
    core_wireframe_pil = create_elastic_wireframe(h, w, base_contours, noise_field_x, noise_field_y, current_core_thickness, solid_color, phase_shift, motion_wisp)

    outer_glow_base = glow_source_pil.filter(ImageFilter.GaussianBlur(radius=current_glow_radius))
    inner_glow_base = glow_source_pil.filter(ImageFilter.GaussianBlur(radius=current_glow_radius / 3))
    
    enhancer = ImageEnhance.Brightness(outer_glow_base)
    outer_glow = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Brightness(inner_glow_base)
    inner_glow = enhancer.enhance(2.0)

    final_image = Image.new("RGBA", (w, h), (*bg_color, 255))
    
    outer_mask = outer_glow.getchannel('A').point(lambda i: i * opacity_wisp)
    inner_mask = inner_glow.getchannel('A').point(lambda i: i * opacity_wisp)
    core_mask = core_wireframe_pil.getchannel('A').point(lambda i: i * opacity_wisp)

    final_image.paste(outer_glow, (0, 0), outer_mask)
    final_image.paste(inner_glow, (0, 0), inner_mask)
    final_image.paste(core_wireframe_pil, (0, 0), core_mask)

    return final_image

def create_elastic_wireframe(h, w, contours, noise_x, noise_y, thickness, color_option, phase_shift, motion_wisp):
    if not contours: return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    displacement_strength = w * 0.15 * (1.0 - motion_wisp)
    deformed_contours = []
    for c in contours:
        c_float = c.astype(np.float32)
        norm_x = c_float[:, 0, 0] / w
        norm_y = c_float[:, 0, 1] / h
        noise_shape = noise_x.shape
        sample_x = np.clip((norm_x * (noise_shape[1] - 1)), 0, noise_shape[1] - 1).astype(int)
        sample_y = np.clip((norm_y * (noise_shape[0] - 1)), 0, noise_shape[0] - 1).astype(int)
        dx = noise_x[sample_y, sample_x]
        dy = noise_y[sample_y, sample_x]
        c_float[:, 0, 0] += dx * displacement_strength
        c_float[:, 0, 1] += dy * displacement_strength
        deformed_contours.append(c_float.astype(np.int32))
    wireframe_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if isinstance(color_option, tuple):
        cv2.drawContours(wireframe_rgba, deformed_contours, -1, (*color_option, 255), thickness=thickness)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, deformed_contours, -1, 255, thickness=thickness)
        y_coords, x_coords = np.where(mask > 0)
        num_pixels = len(y_coords)
        if num_pixels > 0:
            sorted_indices = np.argsort(y_coords)
            ranks = np.arange(num_pixels) / float(num_pixels - 1)
            shifted_ranks = (ranks + phase_shift) % 1.0
            pixel_positions = np.empty_like(ranks)
            pixel_positions[sorted_indices] = shifted_ranks
            angle = pixel_positions * 2 * np.pi
            r = (np.sin(angle) * 127.5 + 127.5).astype(np.uint8)
            g = (np.sin(angle + 2 * np.pi / 3) * 127.5 + 127.5).astype(np.uint8)
            b = (np.sin(angle + 4 * np.pi / 3) * 127.5 + 127.5).astype(np.uint8)
            rgb_colors = np.stack([r, g, b], axis=-1)
            wireframe_rgba[y_coords, x_coords, :3] = rgb_colors
            wireframe_rgba[y_coords, x_coords, 3] = 255
    return Image.fromarray(wireframe_rgba)

if __name__ == '__main__':
    image_path = "iontoad.png"
    output_video_path = "logo_animation.mp4"
    background_color = (0, 0, 0)  # Configurable background. E.g., (10, 20, 40) for dark blue.
    color_mode = (30, 150, 255)#(0, 255, 255)  # None for rainbow, "smart", or an (R, G, B) tuple.
    noise_scale = 10 #Wavelength of noise
    end_screen = 1
    duration_seconds = 2.5
    fps = 60
    total_frames = int((duration_seconds - end_screen) * fps)
    extra_frames = end_screen * fps

    # ---
    print("Step 1: Analyzing image and finding contours...")
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (1920,1920))
    
    h, w, _ = img_cv.shape
    
    blur_kernel_size=9
    canny_low_threshold=50
    canny_high_threshold=150
    min_line_length_ratio=0.03
    max_dot_width_ratio=0.025
    max_dot_height_ratio=0.025
    
    diagonal = np.sqrt(h**2 + w**2)
    min_line_length = int(diagonal * min_line_length_ratio)
    max_dot_width = int(w * max_dot_width_ratio)
    max_dot_height = int(h * max_dot_height_ratio)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_low_threshold, canny_high_threshold)
    all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    approved_contours = []
    for c in all_contours:
        if cv2.arcLength(c, False) > min_line_length:
            _x, _y, cw, ch = cv2.boundingRect(c)
            if cw >= max_dot_width or ch >= max_dot_height:
                approved_contours.append(c)
    
    print("Step 2: Generating self-contained noise field...")
    noise_x_gen = PerlinNoise(seed=0)
    noise_y_gen = PerlinNoise(seed=1)
    
    noise_w, noise_h = 256, 256
    noise_field_x = np.zeros((noise_h, noise_w))
    noise_field_y = np.zeros((noise_h, noise_w))
    
    scale = noise_scale
    for y in range(noise_h):
        for x in range(noise_w):
            noise_field_x[y, x] = noise_x_gen.noise(x / scale, y / scale)
            noise_field_y[y, x] = noise_y_gen.noise(x / scale, y / scale)

    # Resolve "smart" color
    if color_mode == "smart":
        resolved_color = get_smart_color_from_image(image_path)
    else:
        resolved_color = color_mode


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"Step 3: Generating {total_frames} frames...")
    import tqdm
    for i in tqdm.tqdm(range(total_frames)):
        progress = i / (total_frames - 1)
        import math
        wisp_factor = progress

        frame_pil = create_feature_glow_frame(
            frame_dims=(h, w),
            base_contours=approved_contours,
            noise_field_x=noise_field_x,
            noise_field_y=noise_field_y,
            bg_color=background_color,
            wisp=wisp_factor,
            solid_color=resolved_color,
            phase_shift=progress
        )
        frame_rgb_pil = frame_pil.convert("RGB")
        frame_bgr_cv = cv2.cvtColor(np.array(frame_rgb_pil), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr_cv)

    for i in range(extra_frames):
        video_writer.write(frame_bgr_cv)


    video_writer.release()
    print(f"\nSuccess! Final animation saved to {output_video_path}")