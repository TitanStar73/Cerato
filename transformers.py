# Functions that transform cv2 frames to cv2 frames

import cv2
import numpy as np
import progress_func
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from sklearn.cluster import KMeans
import math
import colorsys


def pulse_transform(frame, progress, intensity=0.5, brightness_boost=0.25):
    """
    Creates a pulse effect by scaling and optionally brightening the frame.

    Args:
        frame (np.ndarray): OpenCV BGR frame.
        progress (float): Value from 0 to 1, representing transition progress.
        intensity (float): Maximum scale change (e.g., 0.05 = ±5%).
        brightness_boost (float): Max brightness change at peak of pulse.

    Returns:
        np.ndarray: Transformed BGR frame.
    """
    h, w = frame.shape[:2]

    pulse_val = np.sin(progress * np.pi)  #Instead of using a diff prog func

    scale = 1 + (pulse_val * intensity)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    #Crop
    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    scaled_frame = resized[y_start:y_start+h, x_start:x_start+w]

    if brightness_boost != 0:
        boost = pulse_val * brightness_boost
        hsv = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + boost), 0, 255)
        scaled_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return scaled_frame


def vector_to_angle_deg(vector):
    """
    Convert a movement vector (dx, dy) to angle in degrees,
    consistent with image coordinates (OpenCV convention).

    Args:
        vector (tuple/list): (dx, dy), e.g. (1, 0) = right, (0, 1) = down.

    Returns:
        float: angle in degrees, where 0 = right, 90 = down, 180 = left, 270 = up.
    """
    dx, dy = vector
    angle = math.degrees(math.atan2(dy, dx))  # atan2(y, x)
    return angle % 360


#Legacy text generation

def create_text_overlay(text: str, font_path: str, reference_frame: np.ndarray, color=(255, 255, 255)):
    """
    Creates a BGRA overlay with the given text, scaled to the maximum possible size
    such that both height and width of the text fit within the reference frame.
    The returned overlay is cropped tightly to the bounding box of the text.

    Args:
        text (str): The text to render.
        font_path (str): Path to the .ttf font file.
        reference_frame (np.ndarray): Reference BGR/BGRA image to fit text into.
        color (str or tuple): Either a hex string ('#RRGGBB') or an (R,G,B) tuple.

    Returns:
        np.ndarray: BGRA overlay cropped to the text bounding box.
    """
    # hex -> (r,g,b)
    if isinstance(color, str) and color.startswith("#"):
        color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

    #Convert to RGBA
    color = (*color, 255)

    ref_h, ref_w = reference_frame.shape[:2]


    dummy_img = Image.new("RGBA", (ref_w, ref_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    def get_text_bbox(font):
        try:
            return draw.textbbox((0, 0), text, font=font)  #(x0, y0, x1, y1)
        except AttributeError:
            w, h = font.getsize(text)
            return (0, 0, w, h)

    # Find max font size
    font_size = 10
    while True:
        font = ImageFont.truetype(font_path, font_size)
        bbox = get_text_bbox(font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_w > ref_w or text_h > ref_h:
            font_size -= 1
            break
        font_size += 1


    font = ImageFont.truetype(font_path, font_size)
    bbox = get_text_bbox(font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img_pil = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color)  # shift so text is inside canvas

    overlay = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    return overlay


def create_text_overlay_raw(
    text: str,
    font_path: str,
    reference_frame: np.ndarray,
    color=(255, 255, 255),
    border=None,
    border_size=0.0,   # fraction of text height (e.g. 0.05 = 5%)
    glow=None
):
    """
    Creates a BGRA overlay with the given text, scaled to the maximum possible size
    such that both height and width of the text fit within the reference frame.
    Supports optional border and glow effects.
    The returned overlay is cropped tightly to the bounding box of the text (with padding).

    Args:
        text (str): The text to render.
        font_path (str): Path to the .ttf font file.
        reference_frame (np.ndarray): Reference BGR/BGRA image to fit text into.
        color (str or tuple): Either a hex string ('#RRGGBB') or an (R,G,B) tuple.
        border (str or tuple, optional): Color for text border. None = no border.
        border_size (float, optional): Border thickness as fraction of text height.
                                       0 = no border. Example: 0.05 = 5%.
        glow (float, optional): Glow intensity [0,1]. None or 0 = no glow.

    Returns:
        np.ndarray: BGRA overlay cropped to the text bounding box.
    """
    if text == "":
        text = "EMPTY"


    # Convert color if hex string
    def parse_color(c):
        if isinstance(c, str) and c.startswith("#"):
            return tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
        return c

    color = (*parse_color(color), 255)
    border_color = (*parse_color(border), 255) if border is not None else None

    ref_h, ref_w = reference_frame.shape[:2]


    dummy_img = Image.new("RGBA", (ref_w, ref_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    def get_text_bbox(font, stroke=0):
        try:
            return draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        except AttributeError:
            w, h = font.getsize(text)
            return (0, 0, w, h)


    # Find max font size
    font_size = 10
    while True:
        font = ImageFont.truetype(font_path, font_size)
        bbox = get_text_bbox(font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_w > ref_w or text_h > ref_h:
            font_size -= 1
            break
        font_size += 1


    font = ImageFont.truetype(font_path, font_size)
    bbox = get_text_bbox(font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    abs_border = int(round(border_size * text_h)) if border_size > 0 else 0

    glow_pad = int(math.ceil(font_size * 0.5 * (glow if glow else 0)))
    pad = abs_border + glow_pad

    canvas_w = int(math.ceil(text_w + 2 * pad))
    canvas_h = int(math.ceil(text_h + 2 * pad))

    img_pil = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_pil)


    if glow and glow > 0:
        glow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        glow_draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color)
        blur_radius = int(font_size * 0.25 * glow) + 1
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
        img_pil = Image.alpha_composite(img_pil, glow_layer)


    if border_color and abs_border > 0:
        draw.text((pad - bbox[0], pad - bbox[1]),
                  text, font=font, fill=color,
                  stroke_width=abs_border, stroke_fill=border_color)
    else:
        draw.text((pad - bbox[0], pad - bbox[1]),
                  text, font=font, fill=color)

    # Convert to BGRA
    overlay = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    return overlay

def create_text_overlay(*args, shadow = False, complete_text = None, **kwargs):
    if not shadow:
        return create_text_overlay_raw(*args, **kwargs)

    args_attr = ["text", "font_path", "reference_frame", "color", "border", "border_size", "glow"]
    for i,x in enumerate(args):
        kwargs[args_attr[i]] = x

    border_size = float(kwargs["border_size"]) * 1.5
    kwargs["border_size"] = 0
    text = create_text_overlay_raw(**kwargs)

    kwargs["color"] = (0,0,0)
    shadow = create_text_overlay_raw(**kwargs)

    kwargs["text"] = complete_text
    complete_text_overlay = create_text_overlay_raw(**kwargs)

    shadow = add_overlay_inside_frame(alpha_add(complete_text_overlay, 0), absolute_img_resize(alpha_add(shadow, .85), complete_text_overlay), -1, 0)

    text = add_overlay_inside_frame(alpha_add(complete_text_overlay, 0), absolute_img_resize(text, complete_text_overlay), -1, 0)

    temp = add_overlay(alpha_add(kwargs["reference_frame"], 0), shadow, border_size, border_size)

    return crop_to_fit(add_overlay(temp, text, 0, 0))

def crop_to_fit(frame: np.ndarray) -> np.ndarray:
    """
    Crop a BGRA frame to the smallest bounding box containing all non-transparent pixels.

    Args:
        frame (np.ndarray): Input BGRA image (H, W, 4), dtype=uint8.

    Returns:
        np.ndarray: Cropped BGRA image.
    """
    if frame.shape[2] != 4:
        raise ValueError("Input must be a BGRA frame with 4 channels.")

    alpha = frame[:, :, 3]

    coords = cv2.findNonZero(alpha)

    if coords is None:  # fully transparent image
        return np.zeros((1, 1, 4), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = frame[y:y+h, x:x+w]

    return cropped

def absolute_img_resize(frame: np.ndarray, reference_frame: np.ndarray) -> np.ndarray:
    """
    Resize 'frame' so that both its width and height are <= reference_frame's dimensions.
    Aspect ratio is preserved (no stretching).

    Args:
        frame (np.ndarray): Input BGR or BGRA image.
        reference_frame (np.ndarray): Reference BGR or BGRA image.

    Returns:
        np.ndarray: Resized frame (same channels as input).
    """
    h, w = frame.shape[:2]
    ref_h, ref_w = reference_frame.shape[:2]


    scale = min(ref_w / w, ref_h / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def add_overlay_inside_frame(frame, overlay, x: float, y: float):
    """
    Place overlay on frame at normalized coordinates (x,y).

    Args:
        frame (np.ndarray): BGR or BGRA frame
        overlay (np.ndarray): BGR or BGRA overlay (smaller or equal in size to frame)
        x (float): horizontal coordinate [-1, 1]
        y (float): vertical coordinate [-1, 1] (graph-style: +y = up)

    Returns:
        np.ndarray: Frame with overlay applied (same type as frame)
    """

    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    fc = frame.shape[2]
    oc = overlay.shape[2]

    #Get center
    cx = fw // 2 + int((fw - ow) // 2 * x)
    cy = fh // 2 - int((fh - oh) // 2 * y)  # minus because +y is up

    # Top-left corner of overlay
    x1 = max(cx - ow // 2, 0)
    y1 = max(cy - oh // 2, 0)

    # Bottom-right corner
    x2 = min(x1 + ow, fw)
    y2 = min(y1 + oh, fh)

    # Adjust overlay region (crop if needed at boundaries)
    ox1 = 0 if x1 == cx - ow // 2 else (cx - ow // 2 - x1)
    oy1 = 0 if y1 == cy - oh // 2 else (cy - oh // 2 - y1)
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)


    roi_frame = frame[y1:y2, x1:x2]
    roi_overlay = overlay[oy1:oy2, ox1:ox2]

    if oc == 4:  # Overlay has alpha
        alpha = roi_overlay[:, :, 3:4] / 255.0
        alpha_inv = 1 - alpha
        roi_result = alpha * roi_overlay[:, :, :3] + alpha_inv * roi_frame[:, :, :3]

        if fc == 4:  # Frame also has alpha
            frame[y1:y2, x1:x2, :3] = roi_result.astype(np.uint8)

            # Update alpha channel too
            frame[y1:y2, x1:x2, 3:4] = np.clip(
                (alpha * 255 + alpha_inv * roi_frame[:, :, 3:4]), 0, 255
            ).astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = roi_result.astype(np.uint8)

    return frame



def blend_with_mask(f1: np.ndarray, f2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Combine two BGR frames using an alpha mask.

    Args:
        f1 (np.ndarray): First frame (BGR).
        f2 (np.ndarray): Second frame (BGR).
        mask (np.ndarray): BGRA frame used as mask.
                           - Transparent (alpha=0) -> take from f1
                           - Opaque (alpha=255) -> take from f2

    Returns:
        np.ndarray: Resulting BGR frame.
    """

    if f1.shape[:2] != f2.shape[:2] or f1.shape[:2] != mask.shape[:2]:
        raise ValueError("All frames must have the same width and height")

    alpha = mask[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, None]  # make it 3D for broadcasting

    #Blend
    result = (f1 * (1 - alpha) + f2 * alpha).astype(np.uint8)

    return result


def get_smart_color(image_bgra, saturation_threshold=0.25):
    """
    Analyzes an image to find its dominant color and make it "neon".
    If the image is desaturated, it returns white.

    Args:
        image_bgra (np.ndarray): The BGRA image data from OpenCV.
        saturation_threshold (float): The saturation level below which an image is
                                      considered grayscale (0.0 to 1.0).

    Returns:
        tuple: An (R, G, B) tuple of the determined smart color.
    """
    try:
        alpha_channel = image_bgra[:, :, 3]
        pixels = image_bgra[alpha_channel > 0, :3]

        if len(pixels) == 0:
            return (255, 255, 255)

        #BGR to HSV
        hsv_pixels = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2HSV)[0]
        avg_saturation = np.mean(hsv_pixels[:, 1]) / 255.0


        if avg_saturation < saturation_threshold:
            #print("Smart Color: Image is desaturated. Using white glow.")
            return (255, 255, 255)

        #Dominant Color w K-Means Clustering
        #print("Smart Color: Image is saturated. Finding dominant color...")
        pixels_float = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, center = cv2.kmeans(pixels_float, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_bgr = np.uint8(center[0])

        # Make it "neon"
        dominant_hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        neon_hsv = np.uint8([dominant_hsv[0], 255, 255])
        neon_bgr = cv2.cvtColor(np.uint8([[neon_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

        # Convert to (R, G, B) tuple
        return (int(neon_bgr[2]), int(neon_bgr[1]), int(neon_bgr[0]))

    except Exception as e:
        print(f"Could not determine smart color due to an error: {e}. Defaulting to white.")
        return (255, 255, 255)


def create_angle_gradient_border(alpha_mask, thickness, progress, solid_color):
    """Helper function to create the colored border with progress and color logic."""
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated_mask = cv2.dilate(alpha_mask, kernel, iterations=1)
    border_mask = dilated_mask - alpha_mask

    moments = cv2.moments(alpha_mask)
    if moments["m00"] == 0: # division by zero
        return Image.new("RGBA", (alpha_mask.shape[1], alpha_mask.shape[0]), (0, 0, 0, 0))
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    h, w = alpha_mask.shape
    border_pixels_y, border_pixels_x = np.where(border_mask > 0)
    if border_pixels_y.size == 0:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    angles = np.arctan2(border_pixels_y - cy, border_pixels_x - cx)
    position = ((angles + np.pi) / (2 * np.pi) + 0.25) % 1.0

    visible_indices = np.where(position <= progress)[0]
    if visible_indices.size == 0:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    visible_y, visible_x = border_pixels_y[visible_indices], border_pixels_x[visible_indices]
    visible_positions = position[visible_indices]

    if solid_color:
        rgb_colors = np.array([solid_color] * len(visible_indices), dtype=np.uint8)
    else: # Rainbow logic
        hue = (progress - visible_positions) % 1.0
        rgb_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue])
        rgb_colors = (rgb_colors * 255).astype(np.uint8)

    output_border = np.zeros((h, w, 4), dtype=np.uint8)
    output_border[visible_y, visible_x, :3] = rgb_colors
    output_border[visible_y, visible_x, 3] = border_mask[visible_y, visible_x]

    return Image.fromarray(output_border)

def emissive_glow_frame_neon_tube(frame, overlay, progress=1.0, solid_color=None, border_size=60, glow_radius=120, core_thickness=25):
    """
    Generates a single frame of a realistic, emissive glowing border on an image.

    Args:
        frame (np.ndarray): The background cv2 BGR frame.
        overlay (np.ndarray): The foreground cv2 BGRA frame (the object to make glow).
        solid_color (tuple or str, optional): The color of the glow. Can be an (R, G, B)
                                             tuple, "smart" to auto-detect, or None for
                                             a rainbow effect. Defaults to None.
        border_size (int, optional): The size of the main glow border. Defaults to 45.
        glow_radius (int, optional): The blur radius for the glow effect. Defaults to 90.
        core_thickness (int, optional): The thickness of the sharp inner core. Defaults to 15.
        progress (float, optional): The progress of the glow animation (0.0 to 1.0).
                                    Defaults to 1.0.

    Returns:
        np.ndarray: The final cv2 BGR frame with the glow effect applied.
    """

    final_color = None
    if solid_color == "smart":
        final_color = get_smart_color(overlay)
    elif isinstance(solid_color, (tuple, list)):
        final_color = solid_color


    original_img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay_img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA))

    alpha_mask = overlay[:, :, 3]
    glow_source_border = create_angle_gradient_border(alpha_mask, border_size, progress, final_color)
    sharp_core_border = create_angle_gradient_border(alpha_mask, core_thickness, progress, final_color)


    outer_glow = glow_source_border.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    inner_glow = glow_source_border.filter(ImageFilter.GaussianBlur(radius=glow_radius / 2.5))


    final_image = original_img_pil.copy()
    final_image.paste(outer_glow, (0, 0), outer_glow)
    final_image.paste(inner_glow, (0, 0), inner_glow)
    final_image.paste(inner_glow, (0, 0), inner_glow)
    final_image.paste(sharp_core_border, (0, 0), sharp_core_border)
    final_image.paste(overlay_img_pil, (0, 0), overlay_img_pil)


    final_frame_rgb = np.array(final_image.convert("RGB"))
    final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)

    return final_frame_bgr



def glass_text_effect(frame, overlay, brightness_factor=-0.5):
    f1 = frame
    f2 = add_brightness_universal(frame, brightness_factor)
    return blend_with_mask(f1, f2, overlay)


def whiteout_region(frame: np.ndarray, a: float, b: float, color: tuple = (255, 255, 255), orientation: str = "vertical") -> np.ndarray:
    """
    Take a cv2 BGR frame and fill a region with a given color.

    By default: vertical region (width axis), white color.

    Args:
        frame (np.ndarray): Input BGR frame (H, W, 3).
        a (float): Start coordinate (0 to 1).
        b (float): End coordinate (0 to 1).
        color (tuple): BGR color to fill with (default: white).
        orientation (str): "vertical" (width axis) or "horizontal" (height axis).

    Returns:
        np.ndarray: Frame with region filled.
    """
    h, w = frame.shape[:2]
    a, b = min(a, b), max(a, b)  # ensure order

    if orientation == "vertical":
        x1, x2 = int(a * w), int(b * w)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        frame[:, x1:x2] = color
    elif orientation == "horizontal":
        y1, y2 = int(a * h), int(b * h)
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        frame[y1:y2, :] = color
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    return frame


def add_extra_motion_blur(frame,
                    length_weighted: float = 0.05, # "strength" of the streak in pixels (odd is best)
                    angle_deg: float = 0.0, # direction of motion (0 = to the right, 90 = downward)
                    thickness: float = 2.0, # softness of the streak's width; >1 gives smoother blur
                    iterations: int = 1, # apply kernel multiple times for extra-strong blur
                    border_mode: str = "reflect"  # "reflect" | "replicate" | "wrap" | "constant"
                   ):
    """
    Apply strong directional motion blur to a BGR/BGRA frame.

    Args:
        frame: np.ndarray, dtype uint8 (BGR or BGRA). Output matches input type/channels.
        length: approximate blur length in pixels (will be clamped to >=3 and odd).
        angle_deg: blur direction in degrees (image coords; 0=right, 90=down).
        thickness: controls kernel softness across the motion direction.
        iterations: apply the blur multiple times to increase streak strength.
        border_mode: border handling for convolution.

    Returns:
        np.ndarray: same shape and dtype as input.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if frame.dtype != np.uint8:
        raise TypeError("frame must be uint8")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("frame must be BGR or BGRA (H,W,3/4)")

    h, w, c = frame.shape
    ch = c

    length = length_weighted * (h+w) * .5

    L = max(3, int(round(length)))
    if L % 2 == 0:
        L += 1  # make odd for a centered kernel
    ksz = L
    k = np.zeros((ksz, ksz), np.float32)


    cx = cy = ksz // 2
    theta = np.deg2rad(angle_deg)

    # half-length vector (dx, dy) in image coords (x right, y down)
    dx = np.cos(theta) * (L - 1) / 2.0
    dy = np.sin(theta) * (L - 1) / 2.0

    pt1 = (int(round(cx - dx)), int(round(cy - dy)))
    pt2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(k, pt1, pt2, color=1.0, thickness=max(1, int(round(thickness))), lineType=cv2.LINE_AA)

    # Blur for antialias
    if thickness > 1.0:
        sigma = 0.3 * ((thickness - 1.0) + 0.8) #Scaled
        k = cv2.GaussianBlur(k, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


    s = k.sum()
    if s <= 1e-8:
        k[:] = 0
        cv2.line(k, (0, cy), (ksz - 1, cy), color=1.0, thickness=1, lineType=cv2.LINE_AA)
        s = k.sum()
    k /= s

    # Map border string to OpenCV flag
    border_map = {
        "reflect": cv2.BORDER_REFLECT101,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
        "constant": cv2.BORDER_CONSTANT,
    }
    bflag = border_map.get(border_mode, cv2.BORDER_REFLECT101)


    def convolve_bgr(bgr_img, kernel, iters):
        out = bgr_img
        for _ in range(max(1, int(iters))):
            out = cv2.filter2D(out, ddepth=-1, kernel=kernel, borderType=bflag)
        return out


    dtype = frame.dtype


    if ch == 3: #BGR
        work = frame.copy()

        blurred = convolve_bgr(work, k, iterations)
        return blurred.astype(dtype, copy=False)

    else: # BGRA: premultiply -> blur -> unpremultiply
        b, g, r, a = cv2.split(frame)
        a_f = a.astype(np.float32) / 255.0
        a_f_3 = cv2.merge([a_f, a_f, a_f])

        bgr = cv2.merge([b, g, r]).astype(np.float32)
        bgr_pm = cv2.multiply(bgr, a_f_3)  # premultiply by alpha

        # Blur color and alpha with the same kernel
        bgr_pm_blur = convolve_bgr(bgr_pm, k, iterations)
        a_blur = convolve_bgr(a_f, k, iterations)

        # Unpremultiply (+guard against tiny alpha)
        eps = 1e-6
        inv_a = 1.0 / np.maximum(a_blur, eps)
        inv_a_3 = cv2.merge([inv_a, inv_a, inv_a])
        bgr_unpm = cv2.multiply(bgr_pm_blur, inv_a_3)

        bgr_unpm = np.clip(bgr_unpm, 0, 255).astype(np.uint8)
        a_out = np.clip(a_blur * 255.0, 0, 255).astype(np.uint8)
        out = cv2.merge([bgr_unpm[:, :, 0], bgr_unpm[:, :, 1], bgr_unpm[:, :, 2], a_out])
        return out.astype(dtype, copy=False)


def zoom_object_wise(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Zoom each separate non-transparent object in a BGRA frame individually,
    maintaining their center of mass. The output has the same shape as input.

    Args:
        frame (np.ndarray): Input BGRA image (H, W, 4).
        scale (float): Zoom factor (>1 zooms in, <1 zooms out).

    Returns:
        np.ndarray: BGRA image with zoomed objects.
    """
    assert frame.shape[2] == 4, "Input must be BGRA (4 channels)"
    h, w = frame.shape[:2]

    # Separate alpha channel (non-transparent = objects)
    alpha = frame[:, :, 3]

    # Find connected components in alpha mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8), connectivity=8
    )


    out = np.zeros_like(frame)

    for i in range(1, num_labels):  # background = label 0
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]

        obj = frame[y:y+bh, x:x+bw]
        mask = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8) * 255
        obj = cv2.bitwise_and(obj, obj, mask=mask)


        new_w, new_h = int(bw * scale), int(bh * scale)
        if new_w <= 0 or new_h <= 0:
            continue
        zoomed = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


        dx = int(cx - new_w / 2)
        dy = int(cy - new_h / 2)

        x1, y1 = max(0, dx), max(0, dy)
        x2, y2 = min(w, dx + new_w), min(h, dy + new_h)

        zx1, zy1 = max(0, -dx), max(0, -dy)
        zx2, zy2 = zx1 + (x2 - x1), zy1 + (y2 - y1)


        if x1 < x2 and y1 < y2:
            roi = out[y1:y2, x1:x2]
            obj_roi = zoomed[zy1:zy2, zx1:zx2]

            alpha_obj = obj_roi[:, :, 3:4] / 255.0
            roi[:] = (alpha_obj * obj_roi + (1 - alpha_obj) * roi).astype(np.uint8)

    return out

def zoom_object_wise_adv(frame: np.ndarray, scale_w: float, scale_h) -> np.ndarray:
    """
    Zoom each separate non-transparent object in a BGRA frame individually,
    maintaining their center of mass. The output has the same shape as input.

    Args:
        frame (np.ndarray): Input BGRA image (H, W, 4).
        scale (float): Zoom factor (>1 zooms in, <1 zooms out).

    Returns:
        np.ndarray: BGRA image with zoomed objects.
    """
    assert frame.shape[2] == 4, "Input must be BGRA (4 channels)"
    h, w = frame.shape[:2]

    # Separate alpha channel (non-transparent = objects)
    alpha = frame[:, :, 3]

    # Find connected components in alpha mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8), connectivity=8
    )

    out = np.zeros_like(frame)

    for i in range(1, num_labels):  # background = label 0
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]

        obj = frame[y:y+bh, x:x+bw]
        mask = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8) * 255
        obj = cv2.bitwise_and(obj, obj, mask=mask)

        # Zoom object
        new_w, new_h = int(bw * scale_w), int(bh * scale_h)
        if new_w <= 0 or new_h <= 0:
            continue
        zoomed = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


        dx = int(cx - new_w / 2)
        dy = int(cy - new_h / 2)

        x1, y1 = max(0, dx), max(0, dy)
        x2, y2 = min(w, dx + new_w), min(h, dy + new_h)

        zx1, zy1 = max(0, -dx), max(0, -dy)
        zx2, zy2 = zx1 + (x2 - x1), zy1 + (y2 - y1)


        if x1 < x2 and y1 < y2:
            roi = out[y1:y2, x1:x2]
            obj_roi = zoomed[zy1:zy2, zx1:zx2]

            alpha_obj = obj_roi[:, :, 3:4] / 255.0
            roi[:] = (alpha_obj * obj_roi + (1 - alpha_obj) * roi).astype(np.uint8)

    return out

def fade_to_black(frame, progress):
    """
    Fades the frame to black as progress goes from 0 to 1.
    """
    black_frame = np.zeros_like(frame)
    return cv2.addWeighted(frame, 1 - progress, black_frame, progress, 0)

def fade_to_white(frame, progress):
    """
    Fades the frame to white as progress goes from 0 to 1.
    """
    white_frame = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1 - progress, white_frame, progress, 0)

def inverse_colors(frame, progress):
    """
    Gradually inverts the frame colors as progress goes from 0 to 1.
    """
    inverted = 255 - frame
    return cv2.addWeighted(frame, 1 - progress, inverted, progress, 0)




def radial_blur(bgr_frame, strength=50):
    """
    Apply zoom-based radial blur with correct center alignment. Peaks out at strength = 50 to 100
    """
    h, w = bgr_frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    acc = np.zeros_like(bgr_frame, np.float32)

    zoom_range = 0.15
    zoom_factors = np.linspace(1.0, 1.0 + zoom_range, strength)

    for z in zoom_factors:
        # Build affine matrix for scaling around the true center
        M = np.float32([
            [z, 0, (1 - z) * cx],
            [0, z, (1 - z) * cy]
        ])
        zoomed = cv2.warpAffine(bgr_frame, M, (w, h), flags=cv2.INTER_LINEAR)
        acc += zoomed.astype(np.float32)

    acc /= len(zoom_factors)
    return np.clip(acc, 0, 255).astype(np.uint8)


#For non-quantised strength
def radial_blur(bgr_frame, strength=50.0):
    """
    strength can be float; internally rounded to nearest integer step count.
    """
    h, w = bgr_frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    steps = max(1, int(round(strength)))  # number of zoom samples
    zoom_range = 0.15 * (strength / 20.0)  # scale zoom distance with float strength

    acc = np.zeros_like(bgr_frame, np.float32)
    zoom_factors = np.linspace(1.0, 1.0 + zoom_range, steps)

    for z in zoom_factors:
        M = np.float32([[z, 0, (1 - z) * cx],
                        [0, z, (1 - z) * cy]])
        acc += cv2.warpAffine(bgr_frame, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)

    acc /= steps
    return np.clip(acc, 0, 255).astype(np.uint8)

def create_disc_kernel(radius):
    """Creates a circular 'disc' kernel for blurring."""
    kernel_size = 2 * int(radius) + 1

    # Create a flat, circular structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Normalize the kernel
    kernel = kernel.astype(np.float32) / np.sum(kernel)
    return kernel


def apply_lens_blur(image, strength = 1, blur=20, glow=10.0):
    """
    Applies a realistic lens blur and bokeh effect based on gamma-corrected highlights.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        radius (int): The radius of the disc blur kernel, controls bokeh size.
        gamma (float): The gamma power to isolate bright spots. Higher values create more distinct bokeh.
    """

    radius = int(blur*progress_func.linear(strength) * (min(image.shape[0], image.shape[1])/1000)) #Scaled w.r.t 1000x1000 image
    gamma = (glow*progress_func.cubic(strength)) + 1
    image_float = image.astype(np.float32) / 255.0

    gamma_corrected_image = np.power(image_float, gamma)

    disc_kernel = create_disc_kernel(radius)

    bokeh_effect = cv2.filter2D(gamma_corrected_image, -1, disc_kernel)
    bokeh_effect_corrected = np.power(bokeh_effect, 1.0 / gamma)

    blur_image = cv2.filter2D(image_float, -1, disc_kernel)

    final_image_float = np.maximum(bokeh_effect_corrected, blur_image)


    return (final_image_float * 255).astype(np.uint8)


def camera_shake(frame, progress, intensity_factor=0.5, frequency=1, direction_vector= (1,1)):
    """
    Simulates camera shake effect.

    Args:
        frame (np.ndarray): OpenCV BGR frame.
        progress (float): 0 to 1.
        intensity (int): Max pixel displacement.
        frequency (float): How many shakes total.
    """
    h, w = frame.shape[:2]

    shake_strength = (1 - progress)

    # Offsets from sine waves
    offset_x = int(np.sin(progress * frequency * 2 * np.pi) * shake_strength * w * intensity_factor * direction_vector[0])
    offset_y = int(np.sin(progress * frequency * 2 * np.pi) * shake_strength * h * intensity_factor * direction_vector[1])

    # Create translation matrix
    M = np.float32([[1, 0, offset_x],
                    [0, 1, offset_y]])

    shaken = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return shaken

def zoom_transform(frame, progress, max_zoom=1.5):
    """
    Smoothly zooms into the image from 1x to max_zoom.
    Preserves BGR or BGRA format.

    Args:
        frame (np.ndarray): OpenCV BGR or BGRA frame.
        progress (float): 0 → 1, zoom amount.
        max_zoom (float): Zoom multiplier at progress=1.

    Returns:
        np.ndarray: Zoomed frame (same channel count as input).
    """
    h, w = frame.shape[:2]

    scale = 1 + (max_zoom - 1) * progress
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2

    return resized[y_start:y_start+h, x_start:x_start+w]

def zoom_rotate_transform(frame, progress, theta=15):
    """
    Rotates image by progress * theta degrees and zooms to keep size consistent.

    Args:
        frame (np.ndarray): OpenCV BGR frame.
        progress (float): 0 → 1.
        theta (float): Max rotation angle in degrees.

    Returns:
        np.ndarray: Rotated + zoom-compensated frame.
    """
    h, w = frame.shape[:2]
    angle = theta * progress

    # Scale to avoid black corners after rotation
    radians = np.deg2rad(abs(angle))
    zoom_factor = 1 / (np.cos(radians) - np.sin(radians))
    zoom_factor = max(zoom_factor, 1.0)


    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, zoom_factor)
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

    return rotated

def add_glow_overlay(overlay, glow_color=(255, 0, 255), thickness=10, blur_size=25, intensity=6):
    """
    Add a glowing outline around a BGRA overlay.

    Args:
        bgra (np.ndarray): Input image with alpha (BGRA).
        glow_color (tuple): BGR color of the glow (default purple).
        thickness (int): Outline thickness before blurring.
        blur_size (int): Blur kernel size (controls glow spread).
        intensity (float): Glow brightness factor [0–1].

    Returns:
        np.ndarray: BGRA image with glow added.
    """
    bgra = overlay.copy()

    alpha = bgra[:, :, 3]
    mask = (alpha > 50).astype(np.uint8) * 255


    dilated = cv2.dilate(mask, np.ones((thickness, thickness), np.uint8))
    outline = cv2.subtract(dilated, mask)


    glow = np.zeros_like(bgra, dtype=np.uint8)
    for i in range(3):  # BGR
        glow[:, :, i] = (outline > 0).astype(np.uint8) * glow_color[i]
    glow[:, :, 3] = outline

    glow_blurred = cv2.GaussianBlur(glow, (blur_size, blur_size), blur_size) # Blur to spread glow
    glow_blurred = np.clip(glow_blurred.astype(np.float32) * intensity, 0, 255).astype(np.uint8) # Boost brightness


    result = glow_blurred.copy()

    mask_bool = alpha > 0
    result[mask_bool] = bgra[mask_bool]

    return result

def skin_effect(overlay_raw: np.ndarray, p: float) -> np.ndarray:
    """
    Grows objects in overlay (BGRA) from their center of mass.

    Args:
        overlay (np.ndarray): BGRA image (H, W, 4).
        p (float): Transition progress [0,1].

    Returns:
        np.ndarray: Transformed BGRA image.
    """
    overlay = overlay_raw.copy()

    p = np.clip(p, 0.0, 1.0)

    h, w = overlay.shape[:2]
    result = np.zeros_like(overlay)


    alpha = overlay[:, :, 3]
    mask = (alpha > 50).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):  # skip background (label 0)
        x, y, bw, bh, area = stats[i]
        if area == 0:
            continue

        obj_rgba = overlay[y:y+bh, x:x+bw]
        obj_mask = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8)

        # get COM
        ys, xs = np.nonzero(obj_mask)
        alphas = obj_rgba[ys, xs, 3].astype(float)
        if alphas.sum() == 0:
            continue
        cx = int(np.sum(xs * alphas) / alphas.sum())
        cy = int(np.sum(ys * alphas) / alphas.sum())


        scale = p

        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        scaled = cv2.warpAffine(obj_rgba, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))

        roi = result[y:y+bh, x:x+bw]
        mask_scaled = scaled[:, :, 3] > 0
        roi[mask_scaled] = scaled[mask_scaled]
        result[y:y+bh, x:x+bw] = roi

    return result


def bounce_transform(frame, progress, intensity=0.1, bounces=3):
    """
    Creates a quick 'bouncy' scaling effect.

    Args:
        frame (np.ndarray): OpenCV BGR frame.
        progress (float): 0 → 1.
        intensity (float): Max bounce scale amount.
        bounces (int): Number of bounces within the duration.

    Returns:
        np.ndarray: Bounced frame.
    """
    h, w = frame.shape[:2]

    # Damped sine wave for "bounciness"
    bounce_val = np.sin(progress * np.pi * bounces) * (1 - progress)
    scale = 1 + bounce_val * intensity

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    return resized[y_start:y_start+h, x_start:x_start+w]


def vhs_filter_transform(frame, progress, jitter_intensity=2, color_bleed=2, noise_strength=0.05):
    """
    VHS-like filter with scanlines, color bleeding, and jitter.
    Deterministic based on progress.
    """
    h, w = frame.shape[:2]

    # Seed based on progress
    rng = np.random.default_rng(int(progress * 1e6))


    shift = int(jitter_intensity * np.sin(progress * np.pi * 4))
    jittered = np.roll(frame, shift, axis=1)

    # Color bleed
    b, g, r = cv2.split(jittered)
    b = np.roll(b, color_bleed, axis=1)
    r = np.roll(r, -color_bleed, axis=1)
    bleed_frame = cv2.merge([b, g, r])

    # Scanlines
    scanline_mask = np.tile(((np.arange(h) % 2) * 0.85 + 0.15)[:, None], (1, w))
    bleed_frame = (bleed_frame.astype(np.float32) * scanline_mask[..., None]).astype(np.uint8)

    # Noise
    noise = (rng.random((h, w)) - 0.5) * 255 * noise_strength
    noise = np.repeat(noise[:, :, None], 3, axis=2)
    noisy_frame = np.clip(bleed_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy_frame


def glitch_transform(frame, progress, max_shift=0.02, block_size=0.08):
    """
    RGB split and blocky glitch distortion.
    Deterministic from progress.
    """
    h, w = frame.shape[:2]
    max_shift = int(w * max_shift)
    block_size = int(block_size * h)
    rng = np.random.default_rng(int(progress * 1e6))

    b, g, r = cv2.split(frame)
    shift = int(max_shift * np.sin(progress * np.pi * 2))
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    frame_rgb = cv2.merge([b, g, r])

    # Block shifts
    for y in range(0, h, block_size):
        if rng.random() > 0.7:
            offset = rng.integers(-max_shift, max_shift + 1)
            frame_rgb[y:y+block_size] = np.roll(frame_rgb[y:y+block_size], offset, axis=1)

    return frame_rgb


def glitch_transform(frame, progress, max_shift=0.02, block_size=0.08):
    """
    RGB split and blocky glitch distortion.
    Works for both BGR (3-channel) and BGRA (4-channel).
    Deterministic from progress.
    """
    h, w = frame.shape[:2]
    max_shift = int(w * max_shift)
    block_size = max(1, int(block_size * h))  # ensure at least 1
    rng = np.random.default_rng(int(progress * 1e6))

    has_alpha = frame.shape[2] == 4
    if has_alpha:
        b, g, r, a = cv2.split(frame)
    else:
        b, g, r = cv2.split(frame)


    shift = int(max_shift * np.sin(progress * np.pi * 2))
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)

    frame_rgb = cv2.merge([b, g, r])


    for y in range(0, h, block_size):
        if rng.random() > 0.7:  # deterministic but sparse
            offset = rng.integers(-max_shift, max_shift + 1)
            frame_rgb[y:y+block_size] = np.roll(frame_rgb[y:y+block_size], offset, axis=1)

    if has_alpha:
        return cv2.merge([frame_rgb[:, :, 0],
                               frame_rgb[:, :, 1],
                               frame_rgb[:, :, 2],
                               a])

    return frame_rgb



def film_grain(frame, progress, grain_strength=0.08, flicker_strength=0.05):
    """
    Adds film grain and brightness flicker, deterministic.
    """
    h, w = frame.shape[:2]
    rng = np.random.default_rng(int(progress * 1e6))

    # Grain
    grain = (rng.random((h, w)) - 0.5) * 255 * grain_strength
    grain = np.repeat(grain[:, :, None], 3, axis=2)

    # Flicker
    flicker = 1 + flicker_strength * np.sin(progress * np.pi * 20)

    output = np.clip(frame.astype(np.float32) * flicker + grain, 0, 255).astype(np.uint8)
    return output


def heat_distortion(frame, progress, amplitude=5, frequency=30, speed=2):
    """
    Simulate heat haze distortion.
    - progress: float in [0, 1]
    - amplitude: max pixel offset
    - frequency: spatial frequency of the distortion
    - speed: how fast the waves move
    """
    h, w = frame.shape[:2]
    rng = np.random.default_rng(12345)


    phase_field = rng.random((h, w), dtype=np.float32) * 2 * np.pi


    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    t = progress * speed * 2 * np.pi


    # Distortion mainly horizontal, varying vertically like hot air rising
    offsets_x = amplitude * np.sin((y_indices / frequency) * 2 * np.pi + t + phase_field)
    offsets_y = np.zeros_like(offsets_x)

    # Remap coordinates
    map_x = (x_indices + offsets_x).astype(np.float32)
    map_y = (y_indices + offsets_y).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def chromatic_pulse(frame, progress, strength=0.15):
    """
    Pulses colors with progress.
    """
    b, g, r = cv2.split(frame)
    scale = 1 + strength * np.sin(progress * np.pi * 2)
    r = np.clip(r.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * (2 - scale), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def golden_sequence(n: int, seed: float = 0.1234) -> list[float]:
    """Deterministic low-discrepancy-ish 1-D sequence mapped to [-1, 1]."""
    phi = (math.sqrt(5) - 1) / 2
    u = seed % 1.0
    out = []
    for _ in range(n):
        u = (u + phi) % 1.0
        out.append(2*u - 1)
    return out


def multiple_population(frame, progress, scale=0.4, num_copies=8):
    """
    Covers the image with smaller copies of itself in a deterministic pattern.

    Args:
        frame (np.ndarray): BGR image.
        progress (float): Transition progress [0, 1].
        num_copies (int): Number of smaller copies. Max 8.
        scale (float): Relative size of copies (0.4 = 40% of original).
        pattern_id (int): Used to select deterministic "random" pattern.
    """
    extra_increase = 2.2
    show = 10 #Show only last ___ images

    h, w = frame.shape[:2]
    out = frame.copy()

    num_copies=min(num_copies, 8)

    #Make sure this has as many coords as num_copies
    positions = [(-0.9,-0.4),(-0.6,0.8),(-0.2,-0.9),(0.1,0.3),(0.4,-0.7),(0.7,0.9),(0.9,-0.1),(0.0,0.0)]
    positions = [(x*1.2,y*1.2) for x,y in positions] #Space them out a bit


    for i in range(0, num_copies):
        if progress < i/num_copies:
            break

        if progress > (i + show)/num_copies:
            continue

        effective_scale = scale * (1 + ((i/(num_copies-1))*(extra_increase - 1)))
        tbp = resize_frame(frame.copy(), effective_scale, effective_scale)
        x,y = positions[i]
        x,y = x * (1-scale), y * (1-scale) #Bound it into original scale, switch to effective scale to ALWAYS keep it inside the frame

        if progress < (i+1)/num_copies:
            brightness = i + 1 - progress*num_copies
            if brightness > 1 or brightness < 0:
                raise Exception("bro messed up")

            brightness = brightness**5
            tbp = add_brightness(tbp, brightness)

        out = add_overlay(out, tbp, x, y)

    return out



def bouncy_stripes(frame, progress, num_stripes=8, direction='vertical', glow_intensity=0.4, reveal_end=0.9, bounce_intensity=0.05):
    """
    Stripes reveal with a 'bounce' effect before settling.

    Args:
        frame (np.ndarray): BGR image.
        progress (float): Transition progress [0, 1].
        num_stripes (int): Number of stripes.
        direction (str): 'vertical' or 'horizontal'.
        glow_intensity (float): Brightness boost for glowing stripe.
        reveal_end (float): When all stripes should be fully revealed.
        bounce_intensity (float): Bounce overshoot factor (0.05 = 5% ahead).
    """
    h, w = frame.shape[:2]
    output = np.zeros_like(frame)
    stripe_size = (w // num_stripes) if direction == 'vertical' else (h // num_stripes)

    stripe_progress = progress / reveal_end

    for i in range(num_stripes):
        local_prog = stripe_progress * num_stripes - i
        if local_prog > 0:
            if local_prog < 1:
                # Bounce effect, overshoot then settle
                bounce = local_prog + np.sin(local_prog * np.pi) * bounce_intensity
                bounce = min(bounce, 1)
            else:
                bounce = 1

            if direction == 'vertical':
                x_start = i * stripe_size
                stripe = frame[:, x_start:x_start + stripe_size].copy()
            else:
                y_start = i * stripe_size
                stripe = frame[y_start:y_start + stripe_size, :].copy()

            if bounce < 1:
                stripe = (stripe * bounce).astype(np.uint8)

            if local_prog < 1 and i == int(stripe_progress * num_stripes):
                hsv = cv2.cvtColor(stripe, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + glow_intensity), 0, 255)
                stripe = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            if direction == 'vertical':
                output[:, x_start:x_start + stripe_size] = stripe
            else:
                output[y_start:y_start + stripe_size, :] = stripe

    return output

def cloud_pulse_transform(frame, progress, overlay, MAX_SCALE = 1.5, MAX_ALPHA = .9, MIN_ALPHA = 0):
    current_scale = (progress_func.explosion(progress) * (MAX_SCALE - 1)) + 1
    current_alpha = (MAX_ALPHA - MIN_ALPHA) * (1 - progress_func.linear(progress)) + MIN_ALPHA

    #Find COM
    alpha = overlay[:, :, 3].astype(np.float32)  # alpha channel
    h, w = alpha.shape
    y_indices, x_indices = np.indices((h, w))

    total_alpha = np.sum(alpha)
    if total_alpha == 0:
        # No visible pixels
        return frame.copy()

    XC = np.sum(x_indices * alpha) / total_alpha
    YC = np.sum(y_indices * alpha) / total_alpha


    scaled_w = int(w * current_scale)
    scaled_h = int(h * current_scale)
    overlay_scaled = cv2.resize(overlay, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # alpha adjust
    overlay_scaled_alpha = overlay_scaled[:, :, 3].astype(np.float32) * current_alpha
    overlay_scaled_alpha = np.clip(overlay_scaled_alpha, 0, 255).astype(np.uint8)
    overlay_scaled[:, :, 3] = overlay_scaled_alpha


    scaled_XC = XC * current_scale
    scaled_YC = YC * current_scale

    # Realign COMs
    frame_h, frame_w = frame.shape[:2]
    x_offset = int(XC - scaled_XC)
    y_offset = int(YC - scaled_YC)


    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)

    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(frame_w, x_offset + scaled_w)
    y_end = min(frame_h, y_offset + scaled_h)

    # Corresponding coords from scaled overlay
    overlay_x_start = max(0, -x_offset)
    overlay_y_start = max(0, -y_offset)
    overlay_x_end = overlay_x_start + (x_end - x_start)
    overlay_y_end = overlay_y_start + (y_end - y_start)

    # Paste
    canvas[y_start:y_end, x_start:x_end] = overlay_scaled[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]


    overlay_rgb = canvas[:, :, :3].astype(np.float32)
    overlay_a = (canvas[:, :, 3] / 255.0)[:, :, None]

    frame_rgb = frame.astype(np.float32)
    blended = frame_rgb * (1 - overlay_a) + overlay_rgb * overlay_a

    return blended.astype(np.uint8)


def resize_frame(frame, new_w, new_h):
    if new_h > 1 and new_w == new_h:
        return zoom_transform(frame, 1, new_w)

    return cv2.resize(frame, (int(new_w * frame.shape[1]), int(new_h * frame.shape[0])), interpolation=cv2.INTER_AREA)


def bar_mask(frame, total_bars, current_bars, direction=0):
    h, w = frame.shape[:2]

    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    mask = np.zeros((h, w), dtype=np.uint8)

    if direction in (0, 1):
        bar_height = h // total_bars
        if direction == 0:
            mask[:bar_height*current_bars, :] = 255
        else:  # bottom
            mask[h - bar_height*current_bars:, :] = 255
    else:  # left or right
        bar_width = w // total_bars
        if direction == 2:
            mask[:, :bar_width*current_bars] = 255
        else:  # right
            mask[:, w - bar_width*current_bars:] = 255

    # Apply mask to alpha channel
    frame[:, :, 3] = cv2.bitwise_and(frame[:, :, 3], mask)

    return frame

def _add_overlay_bgra(frame, overlay, x, y):
    """
    Overlay BGRA onto frame (BGR or BGRA).
    x,y = position of CENTER of overlay, normalized
    """

    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]

    # normalized -> pixel offsets
    offset_x = int(x * (w // 2))
    offset_y = int(y * (h // 2))

    # Calculate top-left corner of overlay
    cx_frame, cy_frame = w // 2, h // 2
    cx_overlay, cy_overlay = ow // 2, oh // 2

    top_left_x = cx_frame + offset_x - cx_overlay
    top_left_y = cy_frame + offset_y - cy_overlay

    output = frame.copy()

    # Overlap region
    x1_frame = max(0, top_left_x)
    y1_frame = max(0, top_left_y)
    x2_frame = min(w, top_left_x + ow)
    y2_frame = min(h, top_left_y + oh)

    x1_overlay = max(0, -top_left_x)
    y1_overlay = max(0, -top_left_y)
    x2_overlay = x1_overlay + (x2_frame - x1_frame)
    y2_overlay = y1_overlay + (y2_frame - y1_frame)

    if x1_frame >= x2_frame or y1_frame >= y2_frame:
        return output

    overlay_crop = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Blend
    for c in range(4):
        output[y1_frame:y2_frame, x1_frame:x2_frame, c] = (
            alpha * overlay_crop[:, :, c] +
            alpha_inv * output[y1_frame:y2_frame, x1_frame:x2_frame, c]
        )

    return output


def _add_overlay_hard(frame, overlay, x, y):
    """
    Hard paste overlay (BGR) onto frame (BGR).

    Args:
        frame (np.ndarray): Background image (H, W, 3).
        overlay (np.ndarray): Overlay image (H2, W2, 3).
        x, y (float): Normalized coords for center of overlay:
                      (0,0) = center of frame,
                      (1,1) = top right,
                      (-1,-1) = bottom left,
                      2 or -2 = completely out of frame.

    Returns:
        np.ndarray: Frame with overlay pasted.
    """
    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]

    offset_x = int(x * (w // 2))
    offset_y = int(y * (h // 2))

    cx_frame, cy_frame = w // 2, h // 2
    cx_overlay, cy_overlay = ow // 2, oh // 2

    # Top-left corner of overlay
    top_left_x = cx_frame + offset_x - cx_overlay
    top_left_y = cy_frame + offset_y - cy_overlay

    #Get frame overlay intersection
    x1_frame = max(0, top_left_x)
    y1_frame = max(0, top_left_y)
    x2_frame = min(w, top_left_x + ow)
    y2_frame = min(h, top_left_y + oh)

    x1_overlay = max(0, -top_left_x)
    y1_overlay = max(0, -top_left_y)
    x2_overlay = x1_overlay + (x2_frame - x1_frame)
    y2_overlay = y1_overlay + (y2_frame - y1_frame)

    # If overlay is completely outside frame
    if x1_frame >= x2_frame or y1_frame >= y2_frame:
        return frame.copy()

    # Hard paste
    output = frame.copy()
    output[y1_frame:y2_frame, x1_frame:x2_frame] = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]

    return output

def add_overlay(frame, overlay, x, y):
    """
    x,y -> position of CENTER of overlay
    X,Y -> 1,1 = top right corner, -1,-1 = top right corner
    2 = out of screen
    Both should be the same size?
    """
    if frame.shape[2] == 4:
        return _add_overlay_bgra(frame, overlay, x, y)
    if overlay.shape[2] == 3:
        return _add_overlay_hard(frame, overlay, x, y)


    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]


    offset_x = int(x * (w // 2))
    offset_y = int(y * (h // 2))

    cx_frame, cy_frame = w // 2, h // 2
    cx_overlay, cy_overlay = ow // 2, oh // 2

    top_left_x = cx_frame + offset_x - cx_overlay
    top_left_y = cy_frame + offset_y - cy_overlay


    output = frame.copy()

    # Compute valid region where overlay and frame overlap
    x1_frame = max(0, top_left_x)
    y1_frame = max(0, top_left_y)
    x2_frame = min(w, top_left_x + ow)
    y2_frame = min(h, top_left_y + oh)

    x1_overlay = max(0, -top_left_x)
    y1_overlay = max(0, -top_left_y)
    x2_overlay = x1_overlay + (x2_frame - x1_frame)
    y2_overlay = y1_overlay + (y2_frame - y1_frame)

    if x1_frame >= x2_frame or y1_frame >= y2_frame:
        return output

    # Extract overlay and alpha
    overlay_crop = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Blend
    for c in range(3):
        output[y1_frame:y2_frame, x1_frame:x2_frame, c] = (
            alpha * overlay_crop[:, :, c] +
            alpha_inv * output[y1_frame:y2_frame, x1_frame:x2_frame, c]
        )

    return output

def add_overlay_bbox(frame, overlay, x, y):
    """
    Calculates the bounding box of an overlay on a frame.

    Args:
        frame: The background image.
        overlay: The image to be placed on the frame.
        x: The normalized x-coordinate (-1 to 1) for the center of the overlay.
        y: The normalized y-coordinate (-1 to 1) for the center of the overlay.

    Returns:
        A tuple (x1, y1, x2, y2) representing the top-left and bottom-right
        corners of the bounding box on the frame. If the overlay is
        completely outside the frame, x1 can be >= x2 or y1 >= y2.
    """
    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]


    offset_x = int(x * (w // 2))
    offset_y = int(y * (h // 2))

    #Intended top-left corner of the overlay
    top_left_x = (w // 2) + offset_x - (ow // 2)
    top_left_y = (h // 2) + offset_y - (oh // 2)

    # Get the intersection of the overlay and frame
    x1 = max(0, top_left_x)
    y1 = max(0, top_left_y)
    x2 = min(w, top_left_x + ow)
    y2 = min(h, top_left_y + oh)

    return (x1, y1, x2, y2)


def get_nontransparent_bbox(frame):
    """
    Calculates the bounding box of all non-transparent pixels in a frame.

    Args:
        frame: The input image with an alpha channel (e.g., RGBA).
               The shape should be (height, width, 4).

    Returns:
        A tuple (x1, y1, x2, y2) representing the top-left and bottom-right
        corners of the bounding box. Returns None if the image is fully
        transparent.
    """
    # Check if the image has an alpha channel
    if frame.shape[2] != 4:
        raise ValueError("Input frame must have an alpha channel (e.g., RGBA).")


    alpha_channel = frame[:, :, 3]
    y_coords, x_coords = np.nonzero(alpha_channel)

    if len(x_coords) == 0:
        return None


    x1 = np.min(x_coords)
    y1 = np.min(y_coords)
    x2 = np.max(x_coords) + 1  # Makes range inclusive
    y2 = np.max(y_coords) + 1

    return (int(x1), int(y1), int(x2), int(y2))


def add_blur(frame, method="gaussian", ksize=5):
    """
    Apply blur to a BGR frame using different methods.

    Args:
        frame (np.ndarray): Input BGR image.
        method (str): Blur method -> "average", "gaussian", "median", "bilateral".
        ksize (float): Kernel size [0, .5)

    Returns:
        np.ndarray: Blurred frame.
    """
    h, w = frame.shape[:2]

    # Scale ksize
    ksize = int((min(h, w)/1080) * ksize)
    ksize = 2 * ksize + 1

    if method == "average":
        return cv2.blur(frame, (ksize, ksize))

    elif method == "gaussian":
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    elif method == "median":
        return cv2.medianBlur(frame, ksize)

    elif method == "bilateral":
        return cv2.bilateralFilter(frame, d=ksize, sigmaColor=75, sigmaSpace=75)

    else:
        raise ValueError("Unknown blur method. Use: average, gaussian, median, bilateral.")


def heat_streak_distortion(frame: np.ndarray, direction: str = "vertical",
                           strength: float = 1.0, progress: float = 0.0,
                           streak_size_x: float = 1.0, streak_size_y: float = 1.0, HEAT_SEED = 12345) -> np.ndarray:

    """
    Apply a 'heat distortion with streaks' effect to a BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image (H, W, 3), uint8.
    direction : str, optional
        "vertical" for vertical streaks (default) or "horizontal" for horizontal streaks.
        Allowed values: {"vertical", "horizontal"}.
    strength : float, optional
        Multiplier for distortion strength (default=1.0).
        Recommended range: [0.0, 5.0] (0 = no effect, >3 = heavy distortion).
    progress : float, optional
        Animation progress [0.0, 1.0] for dynamic phase shifting (default=0.0).
        Recommended range: [0.0, 1.0] (loops seamlessly).
    streak_size_x : float, optional
        Multiplier for streak thickness along x (default=1.0).
        Recommended range: [0.2, 5.0] (lower = thinner, higher = broader).
    streak_size_y : float, optional
        Multiplier for streak thickness along y (default=1.0).
        Recommended range: [0.2, 5.0].
    HEAT_SEED : int, optional
        Random seed for deterministic streak patterns (default=42).
        Recommended range: [0, 2**31 - 1].

    Returns
    -------
    np.ndarray
        Output BGR image (H, W, 3), uint8.
    """

    strength = float(strength)
    progress = float(progress)
    streak_size_x = float(streak_size_x)
    streak_size_y = float(streak_size_y)


    BASE_AMP_REL = 0.015
    AMP_Y_SCALE = 0.6

    if direction == "vertical":
        NOISE_SCALE_X_REL = 0.004 * streak_size_x
        NOISE_SCALE_Y_REL = 0.12 * streak_size_y
        ORIENT_AXIS = "y"
    elif direction == "horizontal":
        NOISE_SCALE_X_REL = 0.12 * streak_size_x
        NOISE_SCALE_Y_REL = 0.004 * streak_size_y
        ORIENT_AXIS = "x"
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    WAVE_COUNT = 3
    WAVE_FREQ_MIN_CPI = 1.0
    WAVE_FREQ_MAX_CPI = 6.0
    WAVE_WEIGHT = 0.25
    NOISE_WEIGHT = 1.0 - WAVE_WEIGHT

    CHROMA_SHIFT_REL = 0.002
    BLUR_REL = 0.01
    BRIGHTEN_REL = 0.08
    BORDER_MODE = cv2.BORDER_REFLECT101

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Input must be a BGR image with shape (H, W, 3).")
    h, w = frame.shape[:2]
    m = float(min(h, w))

    def _odd_from_rel(length, rel, minimum=1):
        k = max(minimum, int(round(length * rel)))
        if k % 2 == 0:
            k += 1
        return k

    rng = np.random.default_rng(HEAT_SEED)

    # Base streak noise
    noise = rng.random((h, w), dtype=np.float32)
    kx = _odd_from_rel(w, NOISE_SCALE_X_REL, minimum=3)
    ky = _odd_from_rel(h, NOISE_SCALE_Y_REL, minimum=3)
    streak = cv2.GaussianBlur(noise, (kx, ky), sigmaX=0, sigmaY=0, borderType=BORDER_MODE)
    streak = streak - streak.min()
    streak = (streak / (streak.max() - streak.min() + 1e-6)) * 2.0 - 1.0

    # Low-frequency waves with animated progress
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    waves = np.zeros((h, w), dtype=np.float32)
    coord = yy if ORIENT_AXIS == "y" else xx
    length = h if ORIENT_AXIS == "y" else w
    for _ in range(WAVE_COUNT):
        freq_cpi = rng.uniform(WAVE_FREQ_MIN_CPI, WAVE_FREQ_MAX_CPI)
        freq_cpp = freq_cpi / max(length, 1)
        base_phase = rng.uniform(0, 2*np.pi)

        phase = base_phase + progress * 2*np.pi
        amp = rng.uniform(0.5, 1.0)
        waves += amp * np.sin(2*np.pi * (coord * freq_cpp) + phase)

    if np.max(np.abs(waves)) > 1e-6:
        waves /= np.max(np.abs(waves))

    shape_field = NOISE_WEIGHT * streak + WAVE_WEIGHT * waves

    # Displacement fields
    amp_px = BASE_AMP_REL * strength * m
    dx = shape_field * amp_px
    dy = shape_field * (AMP_Y_SCALE * amp_px)

    # Chromatic shimmer
    chroma_noise = rng.random((3, h, w), dtype=np.float32)
    chroma_noise = (chroma_noise - 0.5) * 2.0
    chroma_amp_x = CHROMA_SHIFT_REL * w
    chroma_amp_y = CHROMA_SHIFT_REL * h

    out_channels = []
    for c in range(3):
        dx_c = dx + chroma_noise[c] * chroma_amp_x * (-1 if c == 0 else (1 if c == 2 else 0.0))
        dy_c = dy + chroma_noise[c] * chroma_amp_y * (-1 if c == 0 else (1 if c == 2 else 0.0))
        map_x = xx + dx_c
        map_y = yy + dy_c
        warped_c = cv2.remap(frame[:, :, c], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=BORDER_MODE)
        out_channels.append(warped_c)
    warped = cv2.merge(out_channels)

    # Local haze
    k_blur = _odd_from_rel(int(m), BLUR_REL, minimum=3)
    blurred = cv2.GaussianBlur(warped, (k_blur, k_blur), sigmaX=0, sigmaY=0, borderType=BORDER_MODE)

    mag = np.sqrt(dx**2 + dy**2)
    if np.max(mag) > 1e-8:
        mask = (mag / (amp_px * np.sqrt(1.0 + AMP_Y_SCALE**2))).astype(np.float32)
    else:
        mask = np.zeros_like(mag, dtype=np.float32)
    mask = np.clip(mask, 0.0, 1.0)
    mask3 = np.dstack([mask, mask, mask])

    warped_f = warped.astype(np.float32) / 255.0
    blurred_f = blurred.astype(np.float32) / 255.0
    mixed = warped_f * (1.0 - mask3) + blurred_f * mask3
    mixed = mixed + mask3 * BRIGHTEN_REL
    mixed = np.clip(mixed, 0.0, 1.0)
    out = (mixed * 255.0 + 0.5).astype(np.uint8)
    return out


def heat_streak_distortion_regional(frame: np.ndarray,
                                          x_min: int, y_min: int, x_max: int, y_max: int,
                                          direction: str = "vertical",
                                          strength: float = 30.0, progress: float = 0.0,
                                          streak_size_x: float = 5, streak_size_y: float = 5,
                                          feather_size: float = 0.08,
                                          HEAT_SEED=10283123) -> np.ndarray:
    """
    Apply a 'heat distortion with streaks' effect to a specific region of a BGR frame,
    allowing the distortion to cohesively "leak" outside the defined bounding box.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image (H, W, 3), uint8.
    x_min, y_min, x_max, y_max : int
        Coordinates defining the bounding box where the distortion originates.
    direction : str, optional
        "vertical" (default) or "horizontal" streaks.
    strength : float, optional
        Multiplier for distortion strength (default=1.0).
    progress : float, optional
        Animation progress [0.0, 1.0] for dynamic phase shifting (default=0.0).
    streak_size_x, streak_size_y : float, optional
        Multipliers for streak thickness (default=1.0).
    feather_size : float, optional
        Relative size of the feathering/blurring applied to the distortion edges (default=0.03).
        A larger value creates a softer, more spread-out "leak".
    HEAT_SEED : int, optional
        Random seed for deterministic patterns (default=20250827).

    Returns
    -------
    np.ndarray
        Output BGR image (H, W, 3), uint8.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Input must be a BGR image with shape (H, W, 3).")

    full_h, full_w = frame.shape[:2]
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    if roi_w <= 0 or roi_h <= 0:
        return frame

    strength = float(strength)
    progress = float(progress)
    streak_size_x = float(streak_size_x)
    streak_size_y = float(streak_size_y)

    BASE_AMP_REL = 0.015
    AMP_Y_SCALE = 0.6

    if direction == "vertical":
        NOISE_SCALE_X_REL = 0.004 * streak_size_x
        NOISE_SCALE_Y_REL = 0.12 * streak_size_y
        ORIENT_AXIS = "y"
    elif direction == "horizontal":
        NOISE_SCALE_X_REL = 0.12 * streak_size_x
        NOISE_SCALE_Y_REL = 0.004 * streak_size_y
        ORIENT_AXIS = "x"
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    WAVE_COUNT = 3
    WAVE_FREQ_MIN_CPI = 1.0
    WAVE_FREQ_MAX_CPI = 6.0
    WAVE_WEIGHT = 0.25
    NOISE_WEIGHT = 1.0 - WAVE_WEIGHT
    CHROMA_SHIFT_REL = 0.002
    BLUR_REL = 0.01
    BRIGHTEN_REL = 0.08
    BORDER_MODE = cv2.BORDER_REFLECT101

    m = float(min(roi_h, roi_w))

    def _odd_from_rel(length, rel, minimum=1):
        k = max(minimum, int(round(length * rel)))
        return k + (1 - k % 2)

    rng = np.random.default_rng(HEAT_SEED)

    #Create distortion pattern
    noise_roi = rng.random((roi_h, roi_w), dtype=np.float32)
    kx_roi = _odd_from_rel(roi_w, NOISE_SCALE_X_REL, minimum=3)
    ky_roi = _odd_from_rel(roi_h, NOISE_SCALE_Y_REL, minimum=3)
    streak_roi = cv2.GaussianBlur(noise_roi, (kx_roi, ky_roi), 0)
    streak_roi = (streak_roi - streak_roi.min()) / (streak_roi.max() - streak_roi.min() + 1e-6) * 2.0 - 1.0

    xx_roi, yy_roi = np.meshgrid(np.arange(roi_w, dtype=np.float32), np.arange(roi_h, dtype=np.float32))
    waves_roi = np.zeros((roi_h, roi_w), dtype=np.float32)
    coord_roi = yy_roi if ORIENT_AXIS == "y" else xx_roi
    length_roi = roi_h if ORIENT_AXIS == "y" else roi_w
    for _ in range(WAVE_COUNT):
        freq_cpi = rng.uniform(WAVE_FREQ_MIN_CPI, WAVE_FREQ_MAX_CPI)
        freq_cpp = freq_cpi / max(length_roi, 1)
        phase = rng.uniform(0, 2 * np.pi) + progress * 2 * np.pi
        amp = rng.uniform(0.5, 1.0)
        waves_roi += amp * np.sin(2 * np.pi * (coord_roi * freq_cpp) + phase)
    if np.max(np.abs(waves_roi)) > 1e-6:
        waves_roi /= np.max(np.abs(waves_roi))

    shape_field_roi = NOISE_WEIGHT * streak_roi + WAVE_WEIGHT * waves_roi
    amp_px = BASE_AMP_REL * strength * m
    dx_roi = shape_field_roi * amp_px
    dy_roi = shape_field_roi * (AMP_Y_SCALE * amp_px)

    #Place ROI patterns into full-frame maps and add feathering
    dx_full = np.zeros((full_h, full_w), dtype=np.float32)
    dy_full = np.zeros((full_h, full_w), dtype=np.float32)
    dx_full[y_min:y_max, x_min:x_max] = dx_roi
    dy_full[y_min:y_max, x_min:x_max] = dy_roi

    # Feather the displacement maps to create the "leaky light" effect (and smoothing)
    k_feather = _odd_from_rel(min(full_h, full_w), feather_size, minimum=3)
    dx_full = cv2.GaussianBlur(dx_full, (k_feather, k_feather), 0)
    dy_full = cv2.GaussianBlur(dy_full, (k_feather, k_feather), 0)


    # Apply the distortion
    xx_full, yy_full = np.meshgrid(np.arange(full_w, dtype=np.float32), np.arange(full_h, dtype=np.float32))
    map_x = xx_full + dx_full
    map_y = yy_full + dy_full

    # Chromatic shimmer
    chroma_noise_roi = (rng.random((3, roi_h, roi_w), dtype=np.float32) - 0.5) * 2.0
    chroma_amp_x = CHROMA_SHIFT_REL * roi_w
    chroma_amp_y = CHROMA_SHIFT_REL * roi_h
    out_channels = []
    for c in range(3):
        dx_c_full = np.zeros_like(dx_full)
        dy_c_full = np.zeros_like(dy_full)
        dx_c_full[y_min:y_max, x_min:x_max] = chroma_noise_roi[c] * chroma_amp_x * (-1 if c == 0 else (1 if c == 2 else 0.0))
        dy_c_full[y_min:y_max, x_min:x_max] = chroma_noise_roi[c] * chroma_amp_y * (-1 if c == 0 else (1 if c == 2 else 0.0))

        dx_c_full = cv2.GaussianBlur(dx_c_full, (k_feather, k_feather), 0)
        dy_c_full = cv2.GaussianBlur(dy_c_full, (k_feather, k_feather), 0)

        warped_c = cv2.remap(frame[:, :, c], map_x + dx_c_full, map_y + dy_c_full, interpolation=cv2.INTER_LINEAR, borderMode=BORDER_MODE)
        out_channels.append(warped_c)
    warped = cv2.merge(out_channels)

    # Haze
    k_blur = _odd_from_rel(int(min(full_h, full_w)), BLUR_REL, minimum=3)
    blurred = cv2.GaussianBlur(warped, (k_blur, k_blur), 0)

    mag_roi = np.sqrt(dx_roi**2 + dy_roi**2)
    mask_roi = (mag_roi / (amp_px * np.sqrt(1.0 + AMP_Y_SCALE**2) + 1e-6)).astype(np.float32)

    mask_full = np.zeros((full_h, full_w), dtype=np.float32)
    mask_full[y_min:y_max, x_min:x_max] = mask_roi
    mask_full = cv2.GaussianBlur(mask_full, (k_feather, k_feather), 0)
    mask_full = np.clip(mask_full, 0.0, 1.0)
    mask3 = np.dstack([mask_full] * 3)

    mixed = warped.astype(np.float32) * (1.0 - mask3) + blurred.astype(np.float32) * mask3
    mixed += mask3 * (BRIGHTEN_REL * 255.0)
    return np.clip(mixed, 0, 255).astype(np.uint8)

def breathe_rotate_jiggle(progress, frame, LOW_ANG=5.1, MAX_ANG=15.0, REVERSE_ALL_ROTATION=False, ZOOM_MID_MULTIPLIER=1.1, MAX_ZOOM_MULTIPLIER=1.6, NUM_BREATHS=1.4, BREATHE_STRENGTH=0.025, BREATHING_DIRECTION_VECTOR=(0, 1.0), FAST_PHASE_PROPORTION=0.3, STILL_POINT_SPLIT=0.3):
    """
    Generates a single transformed frame for a video transition based on a progress value.
    Still at -> p = (1 - FAST_PHASE_PROPORTION) * STILL_POINT_SPLIT -> .21 for current values
    Args:
        progress (float): The progress of the animation, from 0.0 to 1.0.
        frame (np.ndarray): The BGR input image frame.
    Returns:
        np.ndarray: The transformed BGR output frame.
    """

    # Helper functions
    def get_min_zoom(angle_deg, w, h):
        """Calculates the minimum zoom to avoid black borders after rotation."""
        angle_rad = math.radians(angle_deg)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        new_w = w * cos_a + h * sin_a
        new_h = w * sin_a + h * cos_a
        return max(new_w / w, new_h / h)

    def radial_blur(image, blur_strength=0.05):
        """Applies a simple radial blur effect."""
        center_x, center_y = width // 2, height // 2
        blurred_image = image.copy().astype(np.float32)
        if blur_strength <= 0:
            return image
        for i in range(1, 6):
            scale = 1.0 + (i * blur_strength)
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            zoomed = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
            blurred_image = cv2.addWeighted(blurred_image, 0.8, zoomed.astype(np.float32), 0.2, 0)
        return np.clip(blurred_image, 0, 255).astype(np.uint8)


    height, width, _ = frame.shape
    center = (width / 2, height / 2)
    center_vec = np.array([[center[0]], [center[1]]])

    # Normalize breathing vector and find its angle
    breathing_vec = np.array(BREATHING_DIRECTION_VECTOR)
    if np.linalg.norm(breathing_vec) > 0:
        breathing_vec = breathing_vec / np.linalg.norm(breathing_vec)
    breathing_angle_deg = np.degrees(np.arctan2(breathing_vec[1], breathing_vec[0]))

    # Calculate key progress points
    slow_phase_end = 1.0 - FAST_PHASE_PROPORTION
    still_point = slow_phase_end * STILL_POINT_SPLIT

    # lowest zoom level
    base_min_zoom = get_min_zoom(LOW_ANG, width, height)


    is_fast_phase = progress >= slow_phase_end
    blur_amount = 0

    if progress < still_point: # From start to still point
        p_phase = progress / still_point
        angle = np.interp(p_phase, [0, 1], [-LOW_ANG, 0])
        zoom = np.interp(p_phase, [0, 1], [base_min_zoom, 1.0]) # Zoom out
    elif progress < slow_phase_end: # From still point to fast phase
        p_phase = (progress - still_point) / (slow_phase_end - still_point)
        angle = np.interp(p_phase, [0, 1], [0, LOW_ANG])
        zoom = np.interp(p_phase, [0, 1], [1.0, base_min_zoom * ZOOM_MID_MULTIPLIER]) # Zoom in
    else: # Fast phase
        p_phase = (progress - slow_phase_end) / FAST_PHASE_PROPORTION
        angle = np.interp(p_phase, [0, 1], [LOW_ANG, MAX_ANG])
        zoom = np.interp(p_phase, [0, 1], [base_min_zoom * ZOOM_MID_MULTIPLIER, base_min_zoom * MAX_ZOOM_MULTIPLIER])
        blur_amount = np.interp(p_phase, [0, 1], [0, 0.05])

    if REVERSE_ALL_ROTATION:
        angle *= -1


    phase_shift = -still_point * NUM_BREATHS * 2 * np.pi
    breathing_amp = BREATHE_STRENGTH * np.sin(progress * NUM_BREATHS * 2 * np.pi + phase_shift)
    sx_b, sy_b = 1.0 + breathing_amp, 1.0 - breathing_amp


    M_rotate_2x2 = cv2.getRotationMatrix2D((0,0), angle, 1)[:2, :2]
    M_zoom_2x2 = np.array([[zoom, 0], [0, zoom]])

    R_b = cv2.getRotationMatrix2D((0,0), breathing_angle_deg, 1)[:2, :2]
    R_b_inv = cv2.getRotationMatrix2D((0,0), -breathing_angle_deg, 1)[:2, :2]
    S_b = np.array([[sx_b, 0], [0, sy_b]])
    M_breathe_transform = R_b @ S_b @ R_b_inv

    M_total_2x2 = M_rotate_2x2 @ M_zoom_2x2 @ M_breathe_transform
    translation = (np.identity(2) - M_total_2x2) @ center_vec
    M_final = np.hstack((M_total_2x2, translation))

    # Apply
    transformed_img = cv2.warpAffine(frame, M_final, (width, height), borderMode=cv2.BORDER_REFLECT)
    if is_fast_phase:
        transformed_img = radial_blur(transformed_img, blur_strength=blur_amount)

    return transformed_img



def silhouette(img, color=(255, 255, 255), inverse=False):
    alpha = img[:, :, 3]
    mask = (alpha > 50).astype(np.uint8) * 255

    if inverse:
        mask = cv2.bitwise_not(mask)

    out = np.zeros_like(img)

    for i in range(3):  # BGR
        out[:, :, i] = np.where(mask == 255, color[i], 0)

    out[:, :, 3] = mask
    return out

def add_brightness(frame, x: float):
    """
    Increase brightness of a BGR/BGRA frame. Will give an 'over-exposed' look

    Args:
        frame (np.ndarray): Input BGR or BGRA image (uint8).
        x (float): Brightness factor [0,1].
                   0 = no change, 1 = max brightness (all pixels white).

    Returns:
        np.ndarray: Brightness-adjusted image (same shape as input).
    """
    # Clamp
    x = np.clip(x, 0.0, 1.0)

    # Separate alpha channel if present
    if frame.shape[2] == 4:
        bgr = frame[:, :, :3]
        alpha = frame[:, :, 3]
    else:
        bgr = frame
        alpha = None

    # Compute offset (0 → no change, 1 → shift pixels to max white)
    offset = int(round(255 * x))

    # Apply brightness with OpenCV (handles clipping internally)
    bright_bgr = cv2.convertScaleAbs(bgr, alpha=1.0, beta=offset)

    # Recombine alpha channel if present
    if alpha is not None:
        brightened = np.dstack([bright_bgr, alpha])
    else:
        brightened = bright_bgr
    return brightened


def add_brightness_universal(frame, x: float):
    """
    Adjust brightness of a BGR/BGRA frame.
    Positive values -> 'over-exposed' look (toward white).
    Negative values -> 'under-exposed' look (toward black).

    Args:
        frame (np.ndarray): Input BGR or BGRA image (uint8).
        x (float): Brightness factor [-1,1].
                   -1 = completely black,
                    0 = no change,
                    1 = completely white.

    Returns:
        np.ndarray: Brightness-adjusted image (same shape as input).
    """
    # Clamp
    x = np.clip(x, -1.0, 1.0)

    # Separate alpha channel if present
    if frame.shape[2] == 4:
        bgr = frame[:, :, :3]
        alpha = frame[:, :, 3]
    else:
        bgr = frame
        alpha = None

    if x >= 0:
        # Over-exposed look: move values toward 255
        bright_bgr = cv2.addWeighted(bgr, 1 - x, np.full_like(bgr, 255), x, 0)
    else:
        # Under-exposed look: move values toward 0
        factor = 1 + x   # since x is negative, this is (1-|x|)
        bright_bgr = cv2.convertScaleAbs(bgr, alpha=factor, beta=0)


    if alpha is not None:
        brightened = np.dstack([bright_bgr, alpha])
    else:
        brightened = bright_bgr

    return brightened


def crop_safe(frame, reference_frame):
    """
    Resize and crop a frame to match reference_frame dimensions.

    Args:
        frame (np.ndarray): Input BGR or BGRA frame.
        reference_frame (np.ndarray): Reference BGR or BGRA frame.

    Returns:
        np.ndarray: Cropped frame with same channel count as input frame.
    """
    h_ref, w_ref = reference_frame.shape[:2]
    h, w = frame.shape[:2]

    scale = max(w_ref / w, h_ref / h)
    new_w, new_h = int(w * scale), int(h * scale)


    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x_start = (new_w - w_ref) // 2
    y_start = (new_h - h_ref) // 2
    cropped = resized[y_start:y_start + h_ref, x_start:x_start + w_ref]

    return cropped

def shake_transform(frame, progress,
                 intensity=0.25, # now a fraction of frame size
                 randomness=0.15,
                 bounciness=0.75,
                 direction=(1.0, 0.0), # bias direction in [-1,1]
                 angular_intensity=15):
    """
    Simulates a deterministic camera shake effect for a transition.

    Args:
        frame (np.ndarray): Input BGR/BGRA frame.
        progress (float): Transition progress [0, 1].
        intensity (float): Max shake as fraction of frame size (0.05 = 5%).
        randomness (float): Strength of deterministic jitter.
        bounciness (float): Overshoot factor at extremes.
        direction (tuple): Bias direction (x,y) in [-1,1].
        angular_intensity (float): Max angular shake in degrees.

    Returns:
        np.ndarray: BGR frame with shake effect.
    """

    h, w = frame.shape[:2]
    center = (w // 2, h // 2)


    max_dx = intensity * w
    max_dy = intensity * h

    # Oscillation + damping
    osc = np.sin(progress * np.pi * (2 + bounciness * 2))
    decay = (1 - progress) * (1 + bounciness * 0.5)
    base_strength = osc * decay

    # Directional bias
    dir_x, dir_y = direction
    bias_x = dir_x * max_dx * base_strength
    bias_y = dir_y * max_dy * base_strength


    rng = np.random.default_rng(int(progress * 1e6))
    rand_x = rng.uniform(-1, 1) * randomness * max_dx * base_strength
    rand_y = rng.uniform(-1, 1) * randomness * max_dy * base_strength


    dx = bias_x + rand_x
    dy = bias_y + rand_y
    angle = base_strength * rng.uniform(-1, 1) * angular_intensity

    # Affine transform
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[:, 2] += (dx, dy)

    shaken = cv2.warpAffine(frame, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)


    if frame.shape[2] == 4:
        return shaken[:, :, :3]
    return shaken


def alpha_add(frame: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ensure frame has an alpha channel, then scale/assign it.

    Parameters:
        frame (np.ndarray): BGR or BGRA image.
        alpha (float): Alpha factor (0 → fully transparent, 1 → fully opaque).

    Returns:
        np.ndarray: BGRA frame with adjusted/created alpha.
    """
    if frame.ndim != 3:
        raise ValueError("Input frame must have 3 dimensions (H, W, C).")

    if frame.shape[2] == 3:
        b, g, r = cv2.split(frame)
        a = np.full_like(b, int(alpha * 255), dtype=np.uint8)
        frame = cv2.merge((b, g, r, a))

    elif frame.shape[2] == 4:
        new_alpha = frame[:, :, 3].astype(np.float32) * alpha
        frame = frame.copy()
        frame[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)

    else:
        raise ValueError("Frame must have 3 (BGR) or 4 (BGRA) channels.")

    return frame

def solid_color_frame(frame, color):
    """
    Create a solid BGR frame of the given color, matching the size of input frame.

    Args:
        frame (np.ndarray): Input BGR or BGRA frame.
        color (tuple): Color as (B, G, R) or (B, G, R, A).

    Returns:
        np.ndarray: Solid color frame (same type & shape as input).
    """
    h, w = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) > 2 else 3

    color = tuple(color[:channels])

    solid = np.full((h, w, channels), color, dtype=frame.dtype)
    return solid


def motion_blur_temporal(frames: list[np.ndarray]) -> np.ndarray:
    """
    Apply motion blur by averaging multiple consecutive frames.

    Parameters:
        frames (list[np.ndarray]): List of frames (all same shape + dtype).
                                   Example: [prev3, prev2, prev1, current, next1, next2, next3]

    Returns:
        np.ndarray: Single motion-blurred frame.
    """
    if len(frames) == 0:
        raise ValueError("No frames provided.")

    # Convert all frames to float for precision
    acc = np.zeros_like(frames[0], dtype=np.float32)

    for f in frames:
        acc += f.astype(np.float32)

    blurred = (acc / len(frames)).astype(frames[0].dtype)
    return blurred


def wavy_trippy_effect(frame: np.ndarray,
                  progress: float,
                  center: tuple = None,
                  amplitude: float = 150.0,
                  wavelength: float = 500.0,
                  waves: float = .25,
                  decay: float = 0,
                  envelope: str = "pulse",
                  interpolation=cv2.INTER_LINEAR,
                  border_mode=cv2.BORDER_REFLECT) -> np.ndarray:
    """
    Apply a radial ripple effect to a BGR image using cv2.remap.

    Args:
        frame: input BGR image (H, W, 3), dtype uint8 (or convertible).
        progress: float in [0,1] controlling the ripple's animation progress.
                  Typical usage: 0.0 (start) -> 1.0 (end). This affects the
                  phase / amplitude envelope depending on `envelope`.
        center: (x, y) pixel coordinates of ripple center. If None, image center.
        amplitude: maximum pixel displacement (in px) at peak of the envelope.
        wavelength: spatial wavelength (px) of the sinusoidal ripple.
        waves: how many cycles the ripple will advance across progress=0..1.
        decay: radial decay factor (higher -> ripple fades faster with radius).
               If decay==0 -> no decay.
        envelope: one of {"linear","pulse","none"}. Controls amplitude over progress:
                  - "none": amplitude * progress
                  - "linear": amplitude * progress
                  - "pulse": amplitude * (4 * progress * (1-progress)) (peaks at progress=0.5)
        interpolation: cv2 interpolation flag for remap.
        border_mode: cv2 border mode for remap.

    Returns:
        BGR uint8 image (same shape as input).
    """
    if frame is None:
        raise ValueError("frame must be a valid image array")
    if not (0.0 <= progress <= 1.0):
        # clamp but warn by raising ValueError - user requested progress in [0,1]
        raise ValueError("progress must be within [0,1]")

    h, w = frame.shape[:2]
    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    # prepare coordinate grids
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    dx = xv - cx
    dy = yv - cy
    r = np.sqrt(dx*dx + dy*dy)  # radial distance

    # avoid division by zero for radial unit vector
    r_safe = r.copy()
    r_safe[r_safe == 0] = 1.0

    # phase progression: ripple appears to move outward/inward depending on sign
    # We use r/wavelength to make ripple periodic with radius, and progress*waves
    phase = 2.0 * np.pi * (r / float(wavelength) - progress * float(waves))
    sinus = np.sin(phase)

    # amplitude envelope by progress
    if envelope == "pulse":
        amp_factor = 4.0 * progress * (1.0 - progress)  # peaks at progress=0.5
    elif envelope == "linear":
        amp_factor = progress
    elif envelope == "none":
        amp_factor = 1.0
    else:
        amp_factor = progress  # fallback

    # radial decay so ripples fade with distance: exp(-decay * r / max_radius)
    max_radius = np.sqrt((max(cx, w-cx))**2 + (max(cy, h-cy))**2)
    if decay <= 0:
        decay_envelope = 1.0
    else:
        decay_envelope = np.exp(-decay * (r / (max_radius + 1e-9)))

    # final displacement in pixels along the radial direction
    displacement = amplitude * amp_factor * decay_envelope * sinus

    # produce mapping: shift points along radial unit vector by displacement
    map_x = xv + (dx / r_safe) * displacement
    map_y = yv + (dy / r_safe) * displacement

    # where r == 0 (exact center), the direction dx/r is undefined; keep them at center
    # (dx/r_safe handles it, since r_safe==1 there and dx==dy==0 -> no shift)

    # remap requires float32 maps
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # remap and return same dtype as input
    remapped = cv2.remap(frame, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    return remapped
    #damn this took wayy to long to make ;-;


def apply_sheen(frame, progress, strength=0.5, width=0.05):
    """
    Apply a sheen effect sweeping diagonally (bottom-left to top-right).

    Args:
        frame (np.ndarray): Input BGR or BGRA frame.
        progress (float): Value between 0 and 1 controlling sheen position.
        strength (float): Brightness blend strength (0=no effect, 1=full white).
        width (float): Relative width of sheen band (0-1).
    """
    h, w = frame.shape[:2]
    has_alpha = frame.shape[2] == 4 if frame.ndim == 3 else False


    Y, X = np.mgrid[0:h, 0:w]

    # Diagonal line runs from bottom-left (0) to top-right (1)
    diag = (X / w + (1 - Y / h)) / 2.0


    band = np.exp(-((diag - progress) ** 2) / (2 * (width ** 2)))
    band = (band * strength)[:, :, None]

    # Create white highlight (i.e. the sheen)
    highlight = np.ones_like(frame, dtype=np.float32) * 255

    # Blend sheen
    out = frame.astype(np.float32) * (1 - band) + highlight * band
    out = np.clip(out, 0, 255).astype(np.uint8)


    if has_alpha:
        out[:, :, 3] = frame[:, :, 3]

    return out


def hologram_effect(frame: np.ndarray, shift=0.05, scan_intensity=0.3):
    """
    Apply a holographic glitch effect to objects in a BGRA frame.
    Looks like a rainbowy hologram projection (not just outline).

    Args:
        frame (np.ndarray): Input BGRA frame.
        shift (int): Max pixel shift for RGB channel displacement.
        scan_intensity (float): Intensity of horizontal scanline effect.

    Returns:
        np.ndarray: Frame with hologram effect applied.
    """
    h, w, _ = frame.shape
    shift = shift * (h+w) * .5
    b, g, r, a = cv2.split(frame)

    # Extract objects
    mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)[1]


    holo = np.zeros_like(frame)

    # Generate displacement map (wave distortion)
    displacement = (np.sin(np.linspace(0, np.pi*4, h)) * shift).astype(np.float32)
    displacement = np.tile(displacement[:, None], (1, w))


    # Apply displacement per-channel
    for i, channel in enumerate([b, g, r]):
        M = np.float32([[1, 0, (i-1)*shift], [0, 1, 0]])
        shifted = cv2.warpAffine(channel, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Apply wavy distortion
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + displacement).astype(np.float32)
        map_y = map_y.astype(np.float32)

        distorted = cv2.remap(
            shifted, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        distorted = cv2.bitwise_and(distorted, distorted, mask=mask)

        holo[:, :, i] = distorted


    holo[:, :, 3] = a
    scanlines = (np.sin(np.linspace(0, np.pi*40, h)) * 127 + 128).astype(np.uint8)
    scanlines = np.tile(scanlines[:, None], (1, w))
    scanlines = cv2.bitwise_and(scanlines, scanlines, mask=mask)

    for i in range(3):  # apply to BGR only
        holo[:, :, i] = cv2.addWeighted(holo[:, :, i], 1.0, scanlines, scan_intensity, 0)

    return holo


def color_tint_gradient_map(frame, colors=[(96, 23, 255), (190, 42, 251), (254, 246, 174), (254, 180, 165)]):
    """
    Apply gradient map color grading.
    - colors: list of BGR colors (from black→mid→white).
      e.g. [(0,0,0), (255,0,255), (0,255,255)]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Build gradient LUT
    n = len(colors)
    lut = np.zeros((256, 3), dtype=np.uint8)
    stops = np.linspace(0, 1, n)
    for i in range(256):
        t = i / 255.0
        j = np.searchsorted(stops, t) - 1
        j = np.clip(j, 0, n-2)
        t0, t1 = stops[j], stops[j+1]
        c0, c1 = np.array(colors[j]), np.array(colors[j+1])
        f = (t - t0) / (t1 - t0 + 1e-6)
        lut[i] = (c0*(1-f) + c1*f).astype(np.uint8)

    mapped = lut[gray]
    return mapped


def crossfade(frame1, frame2, fade: float):
    """
    Crossfade between two frames.

    Parameters:
        frame1 (np.ndarray): First frame (cv2 BGR).
        frame2 (np.ndarray): Second frame (cv2 BGR).
        fade (float): Fade factor [0,1],
                      0 -> frame1 fully,
                      1 -> frame2 fully,
                      in-between -> blend.

    Returns:
        np.ndarray: Crossfaded frame.
    """

    fade = max(0.0, min(1.0, fade))

    if frame1.shape != frame2.shape:
        raise ValueError("Frames must be the same shape for crossfade.")

    blended = cv2.addWeighted(frame1, 1 - fade, frame2, fade, 0)
    return blended


def detect_depth_motion(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(int(fps * 0.5), total_frames - 1)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    cx, cy = w // 2, h // 2  # image center

    expansion_scores = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        dx = flow[...,0]
        dy = flow[...,1]

        # Grid of coords
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        vx = x - cx
        vy = y - cy

        # Normalize (to avoid weighting by distance too much)
        norm = np.sqrt(vx**2 + vy**2) + 1e-6

        vx = vx.astype(float)
        vy = vy.astype(float)

        vx /= norm
        vy /= norm


        dot = dx * vx + dy * vy

        # Positive = expansion (towards camera), Negative = contraction (away)
        expansion_scores.append(np.mean(dot))

        prev_gray = gray

    cap.release()

    if not expansion_scores:
        return 0

    avg_score = np.mean(expansion_scores)

    if avg_score > 0.1: #Towards camera
        return -1
    elif avg_score < -0.1:
        return +1
    else:
        return 0


AVG_MOTION = {}
def detect_average_motion(video_path: str):
    global AVG_MOTION
    if video_path in AVG_MOTION:
        return AVG_MOTION[video_path]


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frames to analyze (first 0.5s or whole video)
    max_frames = min(int(fps * 0.5), total_frames - 1)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    motion_vectors = []

    fh, fw = None, None

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        dx = np.mean(flow[...,0])
        dy = np.mean(flow[...,1])

        if fh is None:
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        motion_vectors.append((dx, dy))

        prev_gray = gray

    cap.release()


    norm_dz = detect_depth_motion(video_path)
    if not motion_vectors:
        AVG_MOTION[video_path] = (0.0, 0.0, norm_dz)
        return (0.0, 0.0, norm_dz)  # No motion

    # Average over all motion vectors
    avg_dx = np.mean([v[0] for v in motion_vectors])
    avg_dy = np.mean([v[1] for v in motion_vectors])

    # Normalize so that the larger absolute component is 1
    MIN_SPEED = 0.01 #Part of the screen moved/second, e.g. 0.1 -> 10% of the screen moved per second, anything <= will count as 0 motion
    LIM = MIN_SPEED/fps
    if abs(avg_dx) <= LIM * fw and abs(avg_dy) <= LIM * fh: #Miniscule amt < 0
        AVG_MOTION[video_path] = (0.0, 0.0, norm_dz)
        return (0.0, 0.0, norm_dz)

    scale = max(abs(avg_dx), abs(avg_dy))

    norm_dx = avg_dx / scale
    norm_dy = avg_dy / scale


    AVG_MOTION[video_path] = (norm_dx, norm_dy, norm_dz)
    return (norm_dx, norm_dy, norm_dz)




def detect_empty_sides(overlay):
    """
    overlay: BGRA or RGBA image (H, W, 4) with alpha channel.
    Returns: dict { "top": bool, "bottom": bool, "left": bool, "right": bool }
             True = side is completely transparent
    """
    if overlay.shape[2] != 4:
        raise ValueError("Overlay must have an alpha channel (4 channels).")

    h, w, _ = overlay.shape
    alpha = overlay[:, :, 3] 


    sides = {
        "top":    np.all(alpha[0, :] == 0),
        "bottom": np.all(alpha[-1, :] == 0),
        "left":   np.all(alpha[:, 0] == 0),
        "right":  np.all(alpha[:, -1] == 0)
    }


    if not any(sides.values()):
        counts = {
            "top":    np.count_nonzero(alpha[0, :]),
            "bottom": np.count_nonzero(alpha[-1, :]),
            "left":   np.count_nonzero(alpha[:, 0]),
            "right":  np.count_nonzero(alpha[:, -1])
        }
        # find side with the fewest non-transparent pixels
        weakest_side = min(counts, key=counts.get)
        sides[weakest_side] = True

    return sides


def raw_shake(frame, direction_vector):
    """
    Shake the image based on a direction vector.

    Parameters:
        frame (np.ndarray): Input OpenCV BGR image.
        direction_vector (tuple): (x, y) values in [0,1].
                                  (1,1) = fully shifted top-right.
                                  (0,0) = no shift.

    Returns:
        np.ndarray: Shaken image with reflected borders.
    """
    h, w = frame.shape[:2]

    # Full range shift
    dx = int(direction_vector[0] * w *.5)   # right shift up to width
    dy = int(-direction_vector[1] * h *.5)  # up shift up to height

    # translation matrix
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])

    shaken = cv2.warpAffine(
        frame, M, (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    return shaken


def tight_fit_function_runner(frame):
    """
    Crops a BGRA frame to the smallest bounding box containing non-transparent pixels.
    Returns:
        cropped (np.ndarray): Cropped frame
        undoer (function): Restores cropped frame into original shape
        redoer (function): Crops any new frame using the same bounding box
    """
    if frame.shape[2] != 4:
        raise ValueError("Input must be a BGRA frame (4 channels).")

    alpha = frame[:, :, 3]

    ys, xs = np.nonzero(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        def undoer(f):
            return np.zeros_like(frame, dtype=frame.dtype)
        def redoer(f):
            return np.zeros((0, 0, 4), dtype=frame.dtype)
        return np.zeros((0, 0, 4), dtype=frame.dtype), undoer, redoer

    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Crop
    cropped = frame[y_min:y_max+1, x_min:x_max+1, :]

    # Store original shape + crop position for undo/redo
    orig_shape = frame.shape

    def undoer(edited_cropped):
        """
        Restores edited_cropped to original shape with transparent borders.
        """
        restored = np.zeros(orig_shape, dtype=frame.dtype)
        h, w = edited_cropped.shape[:2]
        restored[y_min:y_min+h, x_min:x_min+w, :] = edited_cropped
        return restored

    def redoer(new_frame):
        """
        Crops a new frame using the same bounding box.
        """
        if new_frame.shape != orig_shape:
            raise ValueError("New frame must have the same shape as the original frame.")
        return new_frame[y_min:y_max+1, x_min:x_max+1, :]

    return cropped, undoer, redoer


def tight_fit_function_runner(frame):
    """
    Crops a BGRA frame to the smallest bounding box containing non-transparent pixels.
    A row/column is considered empty if fewer than NON_TRANSPARENT_THRESHOLD of its pixels are non-transparent.

    Returns:
        cropped (np.ndarray): Cropped frame
        undoer (function): Restores cropped frame into original shape
        redoer (function): Crops any new frame using the same bounding box
    """
    if frame.shape[2] != 4:
        raise ValueError("Input must be a BGRA frame (4 channels).")


    # Fraction of pixels in a row/column that must be non-transparent to count as "non-empty"
    NON_TRANSPARENT_THRESHOLD = 0.02  # 2%


    alpha = frame[:, :, 3]
    h, w = alpha.shape


    # mask of non-transparent pixels
    non_transparent = alpha > 0

    row_frac = non_transparent.sum(axis=1) / w
    col_frac = non_transparent.sum(axis=0) / h

    ys = np.where(row_frac >= NON_TRANSPARENT_THRESHOLD)[0]
    xs = np.where(col_frac >= NON_TRANSPARENT_THRESHOLD)[0]

    if len(xs) == 0 or len(ys) == 0:
        def undoer(f):
            return np.zeros_like(frame, dtype=frame.dtype)
        def redoer(f):
            return np.zeros((0, 0, 4), dtype=frame.dtype)
        return np.zeros((0, 0, 4), dtype=frame.dtype), undoer, redoer

    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Crop
    cropped = frame[y_min:y_max+1, x_min:x_max+1, :]

    # Store original shape + crop position for undo/redo
    orig_shape = frame.shape

    def undoer(edited_cropped):
        if edited_cropped is None:
            return None
        """Restores edited_cropped to original shape with transparent borders."""
        restored = np.zeros(orig_shape, dtype=frame.dtype)
        h_new, w_new = edited_cropped.shape[:2]
        restored[y_min:y_min+h_new, x_min:x_min+w_new, :] = edited_cropped
        return restored

    def redoer(new_frame):
        """Crops a new frame using the same bounding box."""
        if new_frame.shape != orig_shape:
            raise ValueError("New frame must have the same shape as the original frame.")
        return new_frame[y_min:y_max+1, x_min:x_min+1, :]

    return cropped, undoer, redoer


def droplet_effect(frame: np.ndarray,
                  progress: float,
                  center: tuple = None,
                  amplitude: float = 0.12,
                  width: float = 0.1,
                  decay: float = 0.0,
                  whiteness: float = "auto",
                  interpolation=cv2.INTER_LINEAR,
                  border_mode=cv2.BORDER_REFLECT) -> np.ndarray:
    """
    Apply a single expanding radial ripple to a BGR or BGRA image,
    with optional fading to white at the ripple.

    Args:
        frame: Input BGR or BGRA image (H, W, 3 or 4), uint8.
        progress: [0,1], controls how far the ripple has expanded.
        center: (x, y) pixel coordinates. Defaults to image center.
        amplitude: Max displacement as fraction of max_dim (0.05 = 5%).
        width: Ripple thickness as fraction of max_dim.
        decay: How much ripple weakens with radius (0 = no decay). -> prop to e^(-decay * (r/R))
        whiteness: [0,1], how much the ripple area should fade into white.
        interpolation: cv2 interpolation flag.
        border_mode: cv2 border mode.

    Returns:
        Image with ripple + whiteness effect.
    """

    if whiteness == "auto":
        whiteness = 1-progress**.75

    if not (0.0 <= progress <= 1.0):
        raise ValueError("progress must be within [0,1]")

    h, w = frame.shape[:2]
    channels = frame.shape[2]
    if channels not in (3, 4):
        raise ValueError("Input frame must have 3 (BGR) or 4 (BGRA) channels")


    max_dim = max(h, w)

    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])


    # coordinate grid
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    dx = xv - cx
    dy = yv - cy
    r = np.sqrt(dx * dx + dy * dy)


    # ripple radius grows with progress
    max_radius = np.sqrt(cx**2 + cy**2)
    ripple_radius = progress * max_radius

    # convert relative params to pixels
    amp_px = amplitude * max_dim
    width_px = width * max_dim

    # Gaussian bump around ripple_radius
    ripple_profile = np.exp(-0.5 * ((r - ripple_radius) / width_px) ** 2)


    if decay > 0:
        ripple_profile *= np.exp(-decay * (r / max_radius))

    displacement = amp_px * ripple_profile


    r_safe = np.where(r == 0, 1, r)
    map_x = xv + (dx / r_safe) * displacement
    map_y = yv + (dy / r_safe) * displacement

    # warp image
    warped = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32),
                       interpolation=interpolation, borderMode=border_mode)

    # whiteness overlay on ripple (a diffraction-like effect)
    if whiteness > 0:
        ripple_strength = (displacement / amp_px).clip(0, 1).astype(np.float32)
        whiteness_mask = (whiteness * ripple_strength)[..., None]  # (H, W, 1)

        if channels == 3:  # BGR
            warped = warped.astype(np.float32)
            white = np.full_like(warped, 255, dtype=np.float32)
            warped = warped * (1 - whiteness_mask) + white * whiteness_mask
            warped = np.clip(warped, 0, 255).astype(np.uint8)

        else:  # BGRA
            rgb = warped[..., :3].astype(np.float32)
            alpha = warped[..., 3:]  # keep alpha intact
            white = np.full_like(rgb, 255, dtype=np.float32)
            rgb = rgb * (1 - whiteness_mask) + white * whiteness_mask
            warped = np.concatenate([rgb, alpha.astype(np.float32)], axis=2)
            warped = np.clip(warped, 0, 255).astype(np.uint8)

    return warped


def len_split_overlay(overlay: np.ndarray):
    """
    Splits a BGRA overlay into two objects (main_obj, secondary_obj).
    """
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3]

    mask = (alpha > 50).astype(np.uint8)
    num_labels, _, _,_ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    return num_labels - 1


def split_overlay(overlay: np.ndarray):
    """
    Splits a BGRA overlay into two objects (main_obj, secondary_obj).
    """
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3]

    mask = (alpha > 50).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)


    num_objects = num_labels - 1

    # Rules described below:
    # Case A: exactly 2 objects already
    if num_objects == 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_main = np.argmax(areas) + 1
        idx_sec = 3 - idx_main  # since only 1 or 2

        main_obj = np.zeros_like(overlay)
        secondary_obj = np.zeros_like(overlay)
        main_obj[labels == idx_main] = overlay[labels == idx_main]
        secondary_obj[labels == idx_sec] = overlay[labels == idx_sec]
        return main_obj, secondary_obj

    # Case B: more than 2 objects -> cluster centroids into 2 groups
    if num_objects > 2:
        coords = np.array([centroids[i] for i in range(1, num_labels)])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
        groups = {0: [], 1: []}
        for i, label in enumerate(kmeans.labels_, start=1):
            groups[label].append(i)

        # Build group overlays
        group_overlays = []
        for g in groups.values():
            temp = np.zeros_like(overlay)
            for idx in g:
                temp[labels == idx] = overlay[labels == idx]
            group_overlays.append(temp)

        # Decide main vs secondary: fewer objects group is main, unless area rule applies
        g_counts = [len(groups[0]), len(groups[1])]
        g_areas = [np.sum(cv2.cvtColor(g[:, :, :3], cv2.COLOR_BGR2GRAY) > 0) for g in group_overlays]

        if g_counts[0] < g_counts[1]:
            main_idx, sec_idx = (0, 1)
        elif g_counts[1] < g_counts[0]:
            main_idx, sec_idx = (1, 0)
        else:
            main_idx, sec_idx = (0, 1) if g_areas[0] >= g_areas[1] else (1, 0)

        # Area override rule
        if g_areas[sec_idx] > 2 * g_areas[main_idx]:
            main_idx, sec_idx = sec_idx, main_idx

        return group_overlays[main_idx], group_overlays[sec_idx]

    # Case C: only 1 object -> split with vertical line through object center
    if num_objects == 1:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return overlay.copy(), np.zeros_like(overlay)

        x_min, x_max = np.min(xs), np.max(xs)
        obj_center_x = (x_min + x_max) // 2

        # Vertical split
        left_mask = (xs <= obj_center_x)
        right_mask = (xs > obj_center_x)

        obj_left = np.zeros_like(overlay)
        obj_right = np.zeros_like(overlay)
        obj_left[ys[left_mask], xs[left_mask]] = overlay[ys[left_mask], xs[left_mask]]
        obj_right[ys[right_mask], xs[right_mask]] = overlay[ys[right_mask], xs[right_mask]]

        # Determine which is main: closer to frame center
        frame_center_x = w // 2
        dist_left = abs(obj_center_x - frame_center_x) if np.any(left_mask) else np.inf
        dist_right = abs(obj_center_x - frame_center_x) if np.any(right_mask) else np.inf

        if dist_left < dist_right:
            return obj_left, obj_right
        elif dist_right < dist_left:
            return obj_right, obj_left
        else:
            # Tie: choose left as main
            return obj_left, obj_right


def bar_overlay_transformer_adv2(frame, padding=0.1, num_bars=3, inter_bar_padding=0.05, bars_visible=1, direction="vertical"):
    """
    Splits an image into bar overlays with transparent padding.

    Args:
        frame (np.ndarray): Input BGR or BGRA image.
        padding (float): Fraction [0,1] of image width/height to pad on each side.
        num_bars (int): Total number of bars to create.
        inter_bar_padding (float): Fraction [0,1] of image width/height as spacing between bars.
        bars_visible (int): How many bars are visible now.
        direction (str): "vertical" or "horizontal".

    Returns:
        (overlay, next_bar) -> Tuple of BGRA images (same size as frame).
                               next_bar is None if all bars are visible.
    """
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    h, w = frame.shape[:2]

    # Compute effective dimensions
    if direction == "vertical":
        total_padding = 2 * padding * w + (num_bars - 1) * inter_bar_padding * w
        bar_width = (w - total_padding) / num_bars
        bar_height = h - 2 * padding * h
    else:  # "horizontal"
        total_padding = 2 * padding * h + (num_bars - 1) * inter_bar_padding * h
        bar_height = (h - total_padding) / num_bars
        bar_width = w - 2 * padding * w


    overlay = np.zeros_like(frame, dtype=np.uint8)
    next_bar = np.zeros_like(frame, dtype=np.uint8) if bars_visible < num_bars else None


    for i in range(num_bars):
        if direction == "vertical":
            x1 = int(padding * w + i * (bar_width + inter_bar_padding * w))
            x2 = int(x1 + bar_width)
            y1 = int(padding * h)
            y2 = int(y1 + bar_height)
        else:  # "horizontal"
            y1 = int(padding * h + i * (bar_height + inter_bar_padding * h))
            y2 = int(y1 + bar_height)
            x1 = int(padding * w)
            x2 = int(x1 + bar_width)

        roi = frame[y1:y2, x1:x2]

        if i < bars_visible:  # visible bars
            overlay[y1:y2, x1:x2] = roi
        elif i == bars_visible and next_bar is not None:  #next bar
            next_bar[y1:y2, x1:x2] = roi

    return overlay, next_bar

def _bar_overlay_transformer_adv_raw(frame, padding=0.1, num_bars=3, inter_bar_padding=0.05, bars_visible=1, direction="vertical", tight_fit=False):
    """
    Splits an image into bar overlays with transparent padding.

    Args:
        frame (np.ndarray): Input BGR or BGRA image.
        padding (float): Fraction [0,1] of image width/height to pad on each side.
        num_bars (int): Total number of bars to create.
        inter_bar_padding (float): Fraction [0,1] of image width/height as spacing between bars.
        bars_visible (int): How many bars are visible now.
        direction (str): "vertical" or "horizontal".
        tight_fit (bool): If True and frame is BGRA, trims fully transparent rows
                          (for horizontal bars) or columns (for vertical bars)
                          before computing bar regions.

    Returns:
        (overlay, next_bar) -> Tuple of BGRA images (same size as frame).
                               next_bar is None if all bars are visible.
    """
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    h, w = frame.shape[:2]


    y1_t, y2_t, x1_t, x2_t = 0, h, 0, w
    if tight_fit:
        alpha = frame[:, :, 3]

        if direction == "horizontal":
            # Ignore fully transparent top/bottom rows
            non_empty_rows = np.where(alpha.max(axis=1) > 0)[0]
            if non_empty_rows.size > 0:
                y1_t, y2_t = non_empty_rows[0], non_empty_rows[-1] + 1

        elif direction == "vertical":
            # Ignore fully transparent left/right columns
            non_empty_cols = np.where(alpha.max(axis=0) > 0)[0]
            if non_empty_cols.size > 0:
                x1_t, x2_t = non_empty_cols[0], non_empty_cols[-1] + 1

    region_h, region_w = y2_t - y1_t, x2_t - x1_t

    # Compute effective dimensions
    if direction == "vertical":
        total_padding = 2 * padding * region_w + (num_bars - 1) * inter_bar_padding * region_w
        bar_width = (region_w - total_padding) / num_bars
        bar_height = region_h - 2 * padding * region_h
    else:  # "horizontal"
        total_padding = 2 * padding * region_h + (num_bars - 1) * inter_bar_padding * region_h
        bar_height = (region_h - total_padding) / num_bars
        bar_width = region_w - 2 * padding * region_w

    overlay = np.zeros_like(frame, dtype=np.uint8)
    next_bar = np.zeros_like(frame, dtype=np.uint8) if bars_visible < num_bars else None

    for i in range(num_bars):
        if direction == "vertical":
            x1 = int(x1_t + padding * region_w + i * (bar_width + inter_bar_padding * region_w))
            x2 = int(x1 + bar_width)
            y1 = int(y1_t + padding * region_h)
            y2 = int(y1 + bar_height)
        else:  # "horizontal"
            y1 = int(y1_t + padding * region_h + i * (bar_height + inter_bar_padding * region_h))
            y2 = int(y1 + bar_height)
            x1 = int(x1_t + padding * region_w)
            x2 = int(x1 + bar_width)

        roi = frame[y1:y2, x1:x2]

        if i < bars_visible:  # visible bars
            overlay[y1:y2, x1:x2] = roi
        elif i == bars_visible and next_bar is not None:  #next bar
            next_bar[y1:y2, x1:x2] = roi

    return overlay, next_bar


def _bar_overlay_transformer_adv_raw_diag(frame, padding=0.1, num_bars=3, inter_bar_padding=0.05, bars_visible=1, direction="horizontal", **kwargs):
    """
    Horizontal diagonal bars (top points right) with correct parallel hypotenuses.
    First/last = triangles, middle = parallelograms.
    """

    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    h, w = frame.shape[:2]

    xp = 1 - ((2*padding) + (num_bars - 1)*inter_bar_padding)
    xp = xp/(num_bars - 1)
    polygons = []
    if bars_visible >= 1:
        polygons.append([(padding, padding), (padding + xp, padding), (padding, 1 - padding)])

    ibp = inter_bar_padding
    for n in range(1, min(bars_visible, num_bars - 1)):
        polygons.append([
            (padding + (ibp + xp) * n, padding),
            (padding + xp + (ibp + xp) * n, padding),
            (padding + (ibp + xp) * n, 1 - padding),
            (padding - xp + (ibp + xp) * n, 1 - padding),
        ])

    if bars_visible == num_bars:
        polygons.append([(1 - padding, padding), (1 - padding, 1 - padding), (1 - padding - xp, 1 - padding)])

    if direction == "vertical":
        polygons = [[(y,x) for x,y in arr] for arr in polygons]

    polygons = [[(x * w,y * h) for x,y in arr] for arr in polygons] #Scale
    pts_list = [np.array(poly, dtype=np.int32).reshape((-1, 1, 2)) for poly in polygons]


    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts_list, 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)


    next_overlay = None

    if bars_visible != num_bars:
        polygons = None
        if bars_visible == 0:
            polygons = [[(padding, padding), (padding + xp, padding), (padding, 1 - padding)]]
        elif bars_visible == num_bars - 1:
            polygons = [[(1 - padding, padding), (1 - padding, 1 - padding), (1 - padding - xp, 1 - padding)]]
        else:
            n = bars_visible
            polygons = [[
                (padding + (ibp + xp) * n, padding),
                (padding + xp + (ibp + xp) * n, padding),
                (padding + (ibp + xp) * n, 1 - padding),
                (padding - xp + (ibp + xp) * n, 1 - padding),
            ]]
        if direction == "vertical":
            polygons = [[(y,x) for x,y in arr] for arr in polygons]
        polygons = [[(x * w,y * h) for x,y in arr] for arr in polygons] #Scale
        pts_list = [np.array(poly, dtype=np.int32).reshape((-1, 1, 2)) for poly in polygons]


        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pts_list, 255)
        next_overlay = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame, next_overlay


def bar_overlay_transformer_adv(*args, tight_fit = False, **kwargs):
    if not tight_fit:
        return _bar_overlay_transformer_adv_raw(*args, **kwargs, tight_fit=False)
    cropped, undoer, _ = tight_fit_function_runner(args[0] if len(args) >= 1 else kwargs["frame"])
    args = list(args)
    if len(args) >= 1:
        args[0] = cropped
    else:
        kwargs["frame"] = cropped
    args = tuple(args)
    return [undoer(x) for x in _bar_overlay_transformer_adv_raw(*args, **kwargs, tight_fit=False)]


def bar_overlay_transformer_adv_diag(*args, tight_fit = False, **kwargs):
    if not tight_fit:
        return _bar_overlay_transformer_adv_raw_diag(*args, **kwargs, tight_fit=False)
    cropped, undoer, _ = tight_fit_function_runner(args[0] if len(args) >= 1 else kwargs["frame"])
    args = list(args)
    if len(args) >= 1:
        args[0] = cropped
    else:
        kwargs["frame"] = cropped
    args = tuple(args)
    return [undoer(x) for x in _bar_overlay_transformer_adv_raw_diag(*args, **kwargs, tight_fit=False)]



#Fix/use later, inter_bar_padding broken
def _redun_bar_overlay_transformer_adv(
    frame,
    padding=0.1,
    num_bars=3,
    inter_bar_padding=0.0,
    bars_visible=1,
    direction="vertical",
    tight_fit=False
):
    """
    Splits an image into bar overlays with transparent padding.

    Args:
        frame (np.ndarray): Input BGR or BGRA image.
        padding (float): Fraction [0,1] of image width/height to pad on each side.
        num_bars (int): Total number of bars to create.
        inter_bar_padding (float): Fraction [0,1] of image width/height as spacing between bars.
        bars_visible (int): How many bars are visible now.
        direction (str): "vertical" or "horizontal".
        tight_fit (bool): If True and frame is BGRA, trims fully transparent rows
                          (for horizontal bars) or columns (for vertical bars)
                          before computing bar regions.

    Returns:
        (overlay, next_bar) -> Tuple of BGRA images (same size as frame).
                               next_bar is None if all bars are visible.
    """
    # Ensure BGRA
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    h, w = frame.shape[:2]

    # --- Tight-fit adjustment ---
    y1_t, y2_t, x1_t, x2_t = 0, h, 0, w
    if tight_fit:
        alpha = frame[:, :, 3]
        if direction == "horizontal":
            non_empty_rows = np.where(alpha.max(axis=1) > 0)[0]
            if non_empty_rows.size > 0:
                y1_t, y2_t = non_empty_rows[0], non_empty_rows[-1] + 1
        elif direction == "vertical":
            non_empty_cols = np.where(alpha.max(axis=0) > 0)[0]
            if non_empty_cols.size > 0:
                x1_t, x2_t = non_empty_cols[0], non_empty_cols[-1] + 1

    region_h, region_w = y2_t - y1_t, x2_t - x1_t

    # Compute bar edges (avoid gaps by distributing rounding error)
    if direction == "vertical":
        total_padding = 2 * padding * region_w + (num_bars - 1) * inter_bar_padding * region_w
        bar_width = (region_w - total_padding) / num_bars
        bar_height = region_h - 2 * padding * region_h

        x_edges = [round(x1_t + padding * region_w + i * (bar_width + inter_bar_padding * region_w)) for i in range(num_bars)]
        x_edges.append(x2_t - round(padding * region_w))  # force last edge to fit

        y1 = int(y1_t + padding * region_h)
        y2 = int(y1 + bar_height)

    else:  # horizontal
        total_padding = 2 * padding * region_h + (num_bars - 1) * inter_bar_padding * region_h
        bar_height = (region_h - total_padding) / num_bars
        bar_width = region_w - 2 * padding * region_w

        y_edges = [round(y1_t + padding * region_h + i * (bar_height + inter_bar_padding * region_h)) for i in range(num_bars)]
        y_edges.append(y2_t - round(padding * region_h))  # force last edge to fit

        x1 = int(x1_t + padding * region_w)
        x2 = int(x1 + bar_width)

    # Create empty transparent overlays (full frame size)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    next_bar = np.zeros_like(frame, dtype=np.uint8) if bars_visible < num_bars else None

    for i in range(num_bars):
        if direction == "vertical":
            x1, x2 = x_edges[i], x_edges[i+1]
            roi = frame[y1:y2, x1:x2]
        else:  # horizontal
            y1, y2 = y_edges[i], y_edges[i+1]
            roi = frame[y1:y2, x1:x2]

        if i < bars_visible:
            overlay[y1:y2, x1:x2] = roi
        elif i == bars_visible and next_bar is not None:
            next_bar[y1:y2, x1:x2] = roi

    return overlay, next_bar


def zoom_transform_movement(frame: np.ndarray, zoom: float, direction: tuple[float, float]) -> np.ndarray:
    """
    Apply zoom + directional crop to a BGR/BGRA frame.

    Args:
        frame (np.ndarray): Input frame (BGR or BGRA).
        zoom (float): Zoom factor (>1 = zoom in, <1 = zoom out).
        direction (tuple[float, float]): (x,y) direction vector in [0,1].
                                         (0,0)=top-left, (0.5,0.5)=center, (1,1)=bottom-right.

    Returns:
        np.ndarray: Transformed frame, same size and type as input.
    """
    direction = ((direction[0] + 1)/2, (1 - direction[1])/2)

    h, w = frame.shape[:2]
    new_w, new_h = int(w * zoom), int(h * zoom)

    zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    max_x = new_w - w
    max_y = new_h - h

    # Top-left corner of crop based on direction vector
    crop_x = int(direction[0] * max_x)
    crop_y = int(direction[1] * max_y)

    cropped = zoomed[crop_y:crop_y + h, crop_x:crop_x + w]

    return cropped


def crossfade_transform(frame1: np.ndarray, frame2: np.ndarray, p: float) -> np.ndarray:
    """
    Crossfades two BGR frames with ratio p.

    Args:
        frame1 (np.ndarray): First frame (BGR).
        frame2 (np.ndarray): Second frame (BGR).
        p (float): Blend ratio in [0,1].
                   0 = only frame1,
                   1 = only frame2.

    Returns:
        np.ndarray: Crossfaded frame (BGR).
    """
    p = max(0.0, min(1.0, p))

    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have the same shape to crossfade.")

    blended = cv2.addWeighted(frame1, 1.0 - p, frame2, p, 0.0)
    return blended



class PerlinNoise:
    """
    Pure Python implementation of classic 2D Perlin noise.
    """
    def __init__(self, seed=0):
        self.seed = seed
        # Generate a permutation table using a seeded random number generator
        rng = np.random.default_rng(seed)

        p = np.arange(256, dtype=int)
        rng.shuffle(p)

        # The table is duplicated to avoid expensive modulo operations
        self.p = np.stack([p, p]).flatten()

        # Pre-defined gradient vectors for the 8 grid directions
        self.g = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]], dtype=float)

    def _fade(self, t):
        """A smoothing function: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t):
        """Linear interpolation"""
        return a + t * (b - a)

    def _grad(self, hash_val, x, y):
        """Selects a gradient vector based on a hash and calculates the dot product."""
        g_vec = self.g[hash_val % 8]
        return g_vec[0] * x + g_vec[1] * y

    def noise(self, x, y):
        """Generates a Perlin noise value for a 2D coordinate."""
        # Find the unit grid cell containing the point
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255

        # Get the fractional part of the coordinate
        xf = x - np.floor(x)
        yf = y - np.floor(y)

        # Calculate the smoothed fractional coordinates
        u = self._fade(xf)
        v = self._fade(yf)

        # Hash coordinates of the 4 grid corners
        p = self.p
        aa = p[X] + Y
        ab = p[X] + Y + 1
        ba = p[X + 1] + Y
        bb = p[X + 1] + Y + 1

        # Calculate the dot product between the gradient and distance vectors
        n00 = self._grad(p[aa], xf, yf)
        n01 = self._grad(p[ab], xf, yf - 1)
        n10 = self._grad(p[ba], xf - 1, yf)
        n11 = self._grad(p[bb], xf - 1, yf - 1)

        # Interpolate the results
        x1 = self._lerp(n00, n10, u)
        x2 = self._lerp(n01, n11, u)
        return self._lerp(x1, x2, v)


def get_smart_color_from_image(img_cv, saturation_threshold=0.2):
    try:
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

def create_feature_glow_frame(original_img_pil, base_contours, noise_field_x, noise_field_y, wisp=1.0, solid_color=None, phase_shift=0.0, glow_source_thickness_ratio=0.002, core_thickness_ratio=0.0002, glow_radius_ratio=0.005, og_cv = None):
    h, w = original_img_pil.height, original_img_pil.width
    diagonal = np.sqrt(h**2 + w**2)
    motion_wisp = ease_out_back(wisp)
    opacity_wisp = wisp**.4

    current_glow_radius = max(1, int(diagonal * glow_radius_ratio))
    current_source_thickness = max(1, int(diagonal * glow_source_thickness_ratio))
    current_core_thickness = max(1, int(diagonal * core_thickness_ratio))

    final_color_option = get_final_color_option(solid_color, og_cv)
    glow_source_pil = create_elastic_wireframe(h, w, base_contours, noise_field_x, noise_field_y, current_source_thickness, final_color_option, phase_shift, motion_wisp)
    core_wireframe_pil = create_elastic_wireframe(h, w, base_contours, noise_field_x, noise_field_y, current_core_thickness, final_color_option, phase_shift, motion_wisp)

    outer_glow_base = glow_source_pil.filter(ImageFilter.GaussianBlur(radius=current_glow_radius))
    inner_glow_base = glow_source_pil.filter(ImageFilter.GaussianBlur(radius=current_glow_radius / 3))

    enhancer = ImageEnhance.Brightness(outer_glow_base)
    outer_glow = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Brightness(inner_glow_base)
    inner_glow = enhancer.enhance(2.0)

    final_image = original_img_pil.copy()

    final_image2 = Image.new('RGBA', (w, h), (0, 0, 0, 0))

    outer_mask = outer_glow.getchannel('A').point(lambda i: i * opacity_wisp)
    inner_mask = inner_glow.getchannel('A').point(lambda i: i * opacity_wisp)

    core_mask = core_wireframe_pil.getchannel('A').point(lambda i: i * opacity_wisp)

    final_image.paste(outer_glow, (0, 0), outer_mask)
    final_image.paste(inner_glow, (0, 0), inner_mask)
    final_image.paste(core_wireframe_pil, (0, 0), core_mask)

    final_image2.paste(outer_glow, (0, 0), outer_mask)
    final_image2.paste(inner_glow, (0, 0), inner_mask)
    final_image2.paste(core_wireframe_pil, (0, 0), core_mask)


    return final_image, final_image2


def get_final_color_option(solid_color_option, img_cv = None):
    if solid_color_option == "smart" and img_cv:
        return get_smart_color_from_image(img_cv)
    return solid_color_option


def create_elastic_wireframe(h, w, contours, noise_x, noise_y, thickness, color_option, phase_shift, motion_wisp):
    if not contours:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    displacement_strength = w * 0.05 * (1.0 - motion_wisp)
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
            # Sort by y-coordinate for consistent gradient direction
            sorted_indices = np.argsort(y_coords)
            ranks = np.empty_like(sorted_indices, dtype=float)
            ranks[sorted_indices] = np.arange(num_pixels, dtype=float) / float(num_pixels - 1)

            # Apply the phase shift for animation
            shifted_ranks = (ranks + phase_shift) % 1.0

            # Convert the rank to an angle for a full HSV color cycle
            angle = shifted_ranks * 2 * np.pi


            r = (np.sin(angle) * 127.5 + 127.5).astype(np.uint8)
            g = (np.sin(angle + 2 * np.pi / 3) * 127.5 + 127.5).astype(np.uint8)
            b = (np.sin(angle + 4 * np.pi / 3) * 127.5 + 127.5).astype(np.uint8)

            rgb_colors = np.stack([r, g, b], axis=-1)

            # Apply the calculated colors to the wireframe
            wireframe_rgba[y_coords, x_coords, :3] = rgb_colors
            wireframe_rgba[y_coords, x_coords, 3] = 255 # Set alpha to fully opaque


    return Image.fromarray(wireframe_rgba)


perlin_noise_glob = None
def create_animated_glow_frame_contour_neon(
    frame_bgr,
    progress,
    blur_kernel_size=9,
    canny_low_threshold=40,
    canny_high_threshold=120,
    min_line_length_ratio=0.03,
    max_dot_width_ratio=0.025,
    max_dot_height_ratio=0.025,
    noise_scale=15.0,
    solid_color=None,
    glow_source_thickness_ratio=0.006,
    core_thickness_ratio=0.0014,
    glow_radius_ratio=0.007,
    phi = 0
):
    """
    Applies an animated elastic glow effect to a single CV2 BGR frame.

    Args:
        frame_bgr (np.ndarray): The input image as a CV2 BGR NumPy array.
        progress (float): The animation progress, from 0.0 to 1.0.
        blur_kernel_size (int): Size of the Gaussian blur kernel for edge detection.
        canny_low_threshold (int): Lower threshold for the Canny edge detector.
        canny_high_threshold (int): Higher threshold for the Canny edge detector.
        min_line_length_ratio (float): Minimum contour length relative to image diagonal.
        max_dot_width_ratio (float): Maximum width of ignored contours relative to image width.
        max_dot_height_ratio (float): Maximum height of ignored contours relative to image height.
        noise_scale (float): Higher noise = less wispiness
        solid_color (tuple, optional): A fixed (R, G, B) color for the glow. Defaults to None (rainbow cycle).
        glow_source_thickness_ratio (float): Thickness of the main glow line relative to image diagonal.
        core_thickness_ratio (float): Thickness of the core wireframe relative to image diagonal.
        glow_radius_ratio (float): Radius of the Gaussian blur for the glow effect relative to image diagonal.
        phi (float): Additional color shift
    Returns:
        np.ndarray: The processed CV2 BGR frame with the glow effect applied.
        output_frame, only_wire_frame
    """
    # Get contours
    h, w, _ = frame_bgr.shape
    diagonal = np.sqrt(h**2 + w**2)
    min_line_length = int(diagonal * min_line_length_ratio)
    max_dot_width = int(w * max_dot_width_ratio)
    max_dot_height = int(h * max_dot_height_ratio)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_low_threshold, canny_high_threshold)
    all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours
    approved_contours = []
    for c in all_contours:
        if cv2.arcLength(c, False) > min_line_length:
            _x, _y, cw, ch = cv2.boundingRect(c)
            if cw >= max_dot_width or ch >= max_dot_height:
                approved_contours.append(c)

    # PN
    global perlin_noise_glob
    if perlin_noise_glob is None:
        noise_x_gen = PerlinNoise(seed=0)
        noise_y_gen = PerlinNoise(seed=1)
        noise_w = noise_h = 256
        noise_field_x = np.zeros((noise_h, noise_w))
        noise_field_y = np.zeros((noise_h, noise_w))

        for y in range(noise_h):
            for x in range(noise_w):
                noise_field_x[y, x] = noise_x_gen.noise(x / noise_scale, y / noise_scale)
                noise_field_y[y, x] = noise_y_gen.noise(x / noise_scale, y / noise_scale)

        perlin_noise_glob = (noise_field_x, noise_field_y)
    else:
        noise_field_x, noise_field_y = perlin_noise_glob



    original_img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    wisp_factor = progress**1.5

    # Use progress to shift the phase of the rainbow color cycle
    phase_shift = progress + phi #Remove for static color phase


    frame_pil, frame_pil2 = create_feature_glow_frame(
        original_img_pil=original_img_pil,
        base_contours=approved_contours,
        noise_field_x=noise_field_x.copy(),
        noise_field_y=noise_field_y.copy(),
        wisp=wisp_factor,
        phase_shift=phase_shift,
        solid_color=solid_color,
        glow_source_thickness_ratio=glow_source_thickness_ratio,
        core_thickness_ratio=core_thickness_ratio,
        glow_radius_ratio=glow_radius_ratio,
        og_cv=frame_bgr
    )

    frame_rgb_pil = frame_pil.convert("RGB")
    frame2_rgba_pil = frame_pil2.convert("RGBA")

    final_frame_bgr = cv2.cvtColor(np.array(frame_rgb_pil), cv2.COLOR_RGB2BGR)
    final_wire_bgra = cv2.cvtColor(np.array(frame2_rgba_pil), cv2.COLOR_RGBA2BGRA)

    return final_frame_bgr, final_wire_bgra


pass