import cv2
import numpy as np
import transformers

def _convert_to_float(img):
    """Converts a uint8 image (0-255) to a float32 image (0.0-1.0)."""
    return img.astype(np.float32) / 255.0

def _convert_to_uint8(img):
    """Converts a float32 image (0.0-1.0) back to a uint8 image (0-255)."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def _adjust_tonal_ranges(img_float, shadows, highlights):
    """Adjusts brightness of shadows and highlights using smooth Gaussian masks."""
    if shadows == 0 and highlights == 0:
        return img_float
    lab = cv2.cvtColor(_convert_to_uint8(img_float), cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0] / 255.0

    shadow_mask = np.exp(-((l_channel - 0.25)**2) / 0.08)
    highlight_mask = np.exp(-((l_channel - 0.75)**2) / 0.08)
    
    l_adjusted = l_channel.copy()
    if shadows != 0: l_adjusted += shadow_mask * shadows
    if highlights != 0: l_adjusted += highlight_mask * highlights
    
    lab[:, :, 0] = np.clip(l_adjusted, 0, 1) * 255
    return _convert_to_float(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

def _adjust_vibrance_saturation(img_float, vibrance, saturation):
    """Adjusts vibrance and saturation in HSV color space."""
    hsv = cv2.cvtColor(_convert_to_uint8(img_float), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0

    if vibrance != 0: sat += sat * (1 - sat) * vibrance
    if saturation != 1.0: sat *= saturation
        
    hsv[:, :, 1] = np.clip(sat * 255, 0, 255)
    return _convert_to_float(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

def _apply_vignette(img_float, strength):
    """Applies a darkening vignette to the image corners."""
    rows, cols = img_float.shape[:2]
    strength = 1.01 - strength
    
    kernel_x = cv2.getGaussianKernel(cols, cols * strength)
    kernel_y = cv2.getGaussianKernel(rows, rows * strength)
    
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.max(kernel)
    
    for i in range(3):
        img_float[:,:,i] *= mask
    return img_float


dones = 1

# --- Filter specific transformer functions (not generalised to progress units, instead time based)

import cv2
import numpy as np
import random
import math
from typing import Tuple

def apply_film_grain(
    frame: np.ndarray, 
    time_s: float,
    # --- Size & Alpha Controls ---
    salt_size: float = 6.4,
    pepper_size: float = 6.4,
    salt_alpha: float = 0.7,
    pepper_alpha: float = 0.7,
    # --- Probability Controls ---
    salt_skip_probability: float = 0.75,
    pepper_skip_probability: float = 0.75,
    scratch_probability: float = 1/(1*30), #Assuming 30 fps, about once every 1 second
    scratch_alpha: float = 0.75,
    # --- Density Controls ---
    salt_density: float = 8e-7,
    pepper_density: float = 8e-7,
    # --- Flicker Control ---
    flicker_frequency: float = 5.0,
    flicker_amplitude: float = 6.5,
    # --- Color Controls ---
    salt_color_range: Tuple[int, int] = (200, 240),
    pepper_color_range: Tuple[int, int] = (15, 45)
    ) -> np.ndarray:
    """
    Applies a film imperfection filter.

    Args:
        frame: The input BGR frame.
        time_s: The current time in seconds, used for seeding random effects.
        salt_size / pepper_size: Perceived size of specks (calibrated for 1080p).
        salt_alpha / pepper_alpha: Transparency of specks.
        salt_skip_probability / pepper_skip_probability: Per-frame chance to have NO specks.
        scratch_probability: Per-frame chance of a vertical scratch appearing.
        scratch_alpha: Transparency of the scratch.
        salt_density / pepper_density: Ratio of specks to pixels (when they appear).
        flicker_frequency / flicker_amplitude: Controls the brightness flicker.
        salt_color_range / pepper_color_range: Color range for specks.

    Returns:
        The BGR frame with the film filter applied.
    """

    BASE_RESOLUTION_HEIGHT = 1080.0
    height, width, _ = frame.shape
    processed_frame = frame.copy()

    seed = int(time_s * 1000)
    random.seed(seed)
    np.random.seed(seed)

    scale_factor = height / BASE_RESOLUTION_HEIGHT
    scaled_salt_size = max(1.0, salt_size * scale_factor)
    scaled_pepper_size = max(1.0, pepper_size * scale_factor)
    scaled_scratch_width = max(1, int(round(1.0 * scale_factor)))


    def generate_random_shape(size_multiplier: float) -> np.ndarray:
        canvas_dim = max(3, int(3 * size_multiplier))
        if canvas_dim % 2 == 0: canvas_dim += 1
        num_pixels = max(1, int(3 * size_multiplier**2))
        canvas = np.zeros((canvas_dim, canvas_dim), dtype=np.uint8)
        center = canvas_dim // 2
        canvas[center, center] = 1
        for _ in range(num_pixels):
            y_coords, x_coords = np.where(canvas == 1)
            neighbors = []
            for y, x in zip(y_coords, x_coords):
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < canvas_dim and 0 <= nx < canvas_dim and canvas[ny, nx] == 0:
                        neighbors.append((ny, nx))
            if not neighbors: break
            ny, nx = random.choice(list(set(neighbors)))
            canvas[ny, nx] = 1
        return canvas



    #Vertical Scratches
    if random.random() < scratch_probability:
        scratch_overlay = np.zeros_like(processed_frame)
        x = random.randint(0, width - 1)
        cv2.line(scratch_overlay, (x, 0), (x, height), (random.randint(100,150),)*3, scaled_scratch_width)
        mask = np.any(scratch_overlay > 0, axis=2)
        if np.any(mask):
            processed_frame[mask] = cv2.addWeighted(processed_frame[mask], 1 - scratch_alpha, scratch_overlay[mask], scratch_alpha, 0)

    #Salt and Pepper Specks
    # Add light specks (salt)
    if random.random() >= salt_skip_probability:
        salt_overlay = np.zeros_like(processed_frame)
        num_salt = int(height * width * salt_density)
        for _ in range(num_salt):
            shape = generate_random_shape(scaled_salt_size)
            s_h, s_w = shape.shape
            if s_h < height and s_w < width:
                y, x = random.randint(0, height - s_h - 1), random.randint(0, width - s_w - 1)
                color = random.randint(*salt_color_range)
                region = salt_overlay[y:y+s_h, x:x+s_w]
                region[shape == 1] = (color, color, color)
        
        mask = np.any(salt_overlay > 0, axis=2)
        if np.any(mask):
            processed_frame[mask] = cv2.addWeighted(processed_frame[mask], 1 - salt_alpha, salt_overlay[mask], salt_alpha, 0)
      
    # Add dark specks (pepper)
    if random.random() >= pepper_skip_probability:
        pepper_overlay = np.zeros_like(processed_frame)
        num_pepper = int(height * width * pepper_density)
        for _ in range(num_pepper):
            shape = generate_random_shape(scaled_pepper_size)
            s_h, s_w = shape.shape
            if s_h < height and s_w < width:
                y, x = random.randint(0, height - s_h - 1), random.randint(0, width - s_w - 1)
                color = random.randint(*pepper_color_range)
                region = pepper_overlay[y:y+s_h, x:x+s_w]
                region[shape == 1] = (color, color, color)

        mask = np.any(pepper_overlay > 0, axis=2)
        if np.any(mask):
            processed_frame[mask] = cv2.addWeighted(processed_frame[mask], 1 - pepper_alpha, pepper_overlay[mask], pepper_alpha, 0)

    #Flicker
    beta = flicker_amplitude * math.sin(2 * math.pi * flicker_frequency * time_s)
    final_frame = cv2.convertScaleAbs(processed_frame, alpha=1.0, beta=beta)

    return final_frame

# --- Arbitary values I tweaked basically... Feel free to change later.
VHS_BASE_STRENGTH = 14e-7 * 0.62 #Basically just salt density
FLICKER_AMPLITUDE = 4
FLICKER_FREQUENCY = 3
SALT_SIZE = 1

N_FACT = 1/3 #Increasing makes salt bigger, but less common. (and vice versa)
SALT_SIZE *= N_FACT
VHS_BASE_STRENGTH /= N_FACT


def dark_vhs_filter(image, strength = 1, FPS = 60):
    global dones
    prog = dones/FPS
    dones += 1
    pepper = 1
    salt = 0
    
    return apply_film_grain(image, prog, pepper_density=VHS_BASE_STRENGTH * pepper * strength, salt_density=VHS_BASE_STRENGTH * salt * strength, flicker_amplitude=FLICKER_AMPLITUDE, flicker_frequency=FLICKER_FREQUENCY, salt_size=6.4 * SALT_SIZE)

def light_vhs_filter(image, strength = 1, FPS = 60):
    global dones
    prog = dones/FPS
    dones += 1
    pepper = 0
    salt = 1
    
    return apply_film_grain(image, prog, pepper_density=VHS_BASE_STRENGTH * pepper * strength, salt_density=VHS_BASE_STRENGTH * salt * strength, flicker_amplitude=FLICKER_AMPLITUDE, flicker_frequency=FLICKER_FREQUENCY, salt_size=6.4 * SALT_SIZE)


def glitch_filter(image, strength = 1, FPS = 60):
    global dones
    prog = ((dones * (30/FPS))%10)/10
    dones += 1
    
    return transformers.glitch_transform(image, prog, 0.01*strength, 0.08)

def film_grain_filter(image, strength = 1, FPS = 60):
    global dones
    prog = ((dones * (30/FPS))%10)/10
    dones += 1
    
    return transformers.film_grain(image, prog, 0.08*strength, 0.05*strength)



def apply_borderless_effect(
    bgr_frame: np.ndarray, 
    corner_ratio: float = 0.04, 
    fade_ratio: float = 0.02
) -> np.ndarray:
    """
    Applies rounded corners and a soft, fading edge vignette to a BGR image 
    to create a "borderless screen" effect.

    Args:
        bgr_frame (np.ndarray): The input image in BGR format.
        corner_ratio (float): The radius of the corners as a fraction of the 
                              image's smaller dimension. Default is 0.08.
        fade_ratio (float): The width of the fade/vignette effect as a 
                            fraction of the image's smaller dimension. 
                            Default is 0.07.

    Returns:
        np.ndarray: The processed BGR image with the applied effects.
    """

    #Get radius
    h, w = bgr_frame.shape[:2]
    smaller_dim = min(h, w)
    corner_radius = int(smaller_dim * corner_ratio)
    fade_width = int(smaller_dim * fade_ratio)

    # Ensure fade_width is not larger than corner_radius to avoid artifacts
    fade_width = min(fade_width, corner_radius)


    mask = np.zeros((h, w), dtype=np.uint8)
    
    rect_x1, rect_y1 = fade_width, fade_width
    rect_x2, rect_y2 = w - fade_width, h - fade_width
    inner_radius = corner_radius - fade_width

    # Draw the fully opaque, rounded rectangle center
    if inner_radius > 0:
        cv2.rectangle(mask, (rect_x1 + inner_radius, rect_y1), 
                      (rect_x2 - inner_radius, rect_y2), 255, -1)
        cv2.rectangle(mask, (rect_x1, rect_y1 + inner_radius), 
                      (rect_x2, rect_y2 - inner_radius), 255, -1)
        cv2.circle(mask, (rect_x1 + inner_radius, rect_y1 + inner_radius), 
                   inner_radius, 255, -1)
        cv2.circle(mask, (rect_x2 - inner_radius, rect_y1 + inner_radius), 
                   inner_radius, 255, -1)
        cv2.circle(mask, (rect_x1 + inner_radius, rect_y2 - inner_radius), 
                   inner_radius, 255, -1)
        cv2.circle(mask, (rect_x2 - inner_radius, rect_y2 - inner_radius), 
                   inner_radius, 255, -1)

    else:
        cv2.rectangle(mask, (rect_x1, rect_y1), (rect_x2, rect_y2), 255, -1)


    # The blur kernel size should be odd and proportional to the fade_width
    if fade_width > 0:
        blur_amount = fade_width * 2 + 1
        mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)


    bgra_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
    
    bgra_frame[:, :, 3] = mask
    final_image = np.zeros((h, w, 3), dtype=np.uint8)

    alpha = bgra_frame[:, :, 3] / 255.0
    for c in range(0, 3):
        final_image[:, :, c] = (alpha * bgra_frame[:, :, c] + (1 - alpha) * final_image[:, :, c])

    return final_image


from scipy.interpolate import RegularGridInterpolator

def load_cube_lut(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip headers
    data = []
    for line in lines:
        if line.startswith("#") or line.startswith("TITLE") or line.startswith("LUT_3D_SIZE"):
            continue
        parts = line.strip().split()
        if len(parts) == 3:
            data.append([float(p) for p in parts])
    
    lut = np.array(data)  # shape: (N^3, 3)
    lut = lut[..., ::-1]  # RGB LUT -> BGR LUT
    
    lut_size = round(len(lut) ** (1/3))
    lut = lut.reshape((lut_size, lut_size, lut_size, 3))

    x = y = z = np.linspace(0, 1, lut_size)
    interp = RegularGridInterpolator((x, y, z), lut)

    print(f"Loaded lut {path} with size {lut_size}")

    return interp


def apply_lut(image, lut_interp):
    """must be applied after load_cube_lut"""

    img = image.copy()/255.0

    # Load LUT
    if type(lut_interp) == str:
        raise Exception("Expects interp obj, which can be procured from load_cube_lut(path)")
    
    # Create interpolation function

    # Apply LUT
    shape = img.shape
    flat_img = img.reshape(-1, 3)
    out = lut_interp(flat_img)
    out_img = out.reshape(shape)
    out_img = (out_img * 255).astype(np.uint8)

    # Save result
    return out_img


def _apply_contour(image,prog):
    time_beat = 0.65
    if prog < 1 - time_beat:
        prog = 1
    else:
        prog = (1-prog)/time_beat

    return transformers.create_animated_glow_frame_contour_neon(
        image,
        prog,
        blur_kernel_size=9,
        glow_radius_ratio=0.007,
        solid_color=None ##None (rainbow), "smart", or (r,g,b) form
    )

def apply_contour_main(image, prog):
    return _apply_contour(image, prog)[0]

def apply_contour_plain(image, prog):
    contour = _apply_contour(image, prog)[1]
    background = transformers.solid_color_frame(image, (44,44,44))
    return transformers.add_overlay(background, contour, 0, 0)

#Filters: dark_vhs, light_vhs, glitch, film_grain -> all include strength
#CC: cc_1 -> custom, cc_main -> template (w/ strength)



def cc_1(image, vignette_strength=0.02, highlights=-0.05, vibrance=0.74, contour_mode = 0, progress = -1):
    """
    Applies Vignette, Highlights, and Vibrance adjustments to a cv2 BGR image.

    This function uses the same high-quality algorithms as the web application
    to ensure matching results.

    Args:
        image (np.ndarray): The input image in cv2's BGR format (uint8).
        vignette_strength (float): Controls the vignette intensity.
            - Range: 0.0 (strongest effect) to 1.0 (no effect).
        highlights (float): Adjusts the brightness of the highlights.
            - Range: -1.0 (darkens highlights) to 1.0 (brightens them).
        vibrance (float): Intelligently boosts less-saturated colors.
            - Range: 0.0 (no effect) to 1.0 (strong boost).

    Returns:
        np.ndarray: The edited image in cv2's BGR format (uint8), ready to
                    be saved or displayed.
    """
    # Start the high-precision float workflow
    img_float = _convert_to_float(image)
    
    #Apply Highlights adjustment (shadows are set to 0 for no effect)
    img_float = _adjust_tonal_ranges(img_float, shadows=0.0, highlights=highlights)

    #Apply Vibrance adjustment (saturation is set to 1.0 for no effect)
    img_float = _adjust_vibrance_saturation(img_float, vibrance=vibrance, saturation=1.0)
    
    #Apply Vignette effect if strength is less than 1.0
    if vignette_strength < 1.0:
        img_float = _apply_vignette(img_float, strength=vignette_strength)

    #Convert back to standard 8-bit format before returning
    final_image = _convert_to_uint8(img_float)

    if contour_mode <= 0.5:
        return final_image
    if contour_mode < 1.5:
        return apply_contour_main(final_image, prog=progress)
    return apply_contour_plain(final_image, prog = progress)


def cc_main(image, strength = 1, contour_mode = 0, progress = -1):
    """Accepts -ve values"""
    return cc_1(image, vignette_strength=0.02*max(0, strength), highlights=-0.05*strength, vibrance=0.74*strength, contour_mode=contour_mode, progress=progress)
