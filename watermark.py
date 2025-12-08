import numpy as np
from PIL import Image, ImageFilter
import cv2
from settings import FPS

class Blank():
    def __init__(self, img):
        self.img = img

    def __add__(self,other):
        if type(other) in {Glow, Blank}:
            return Blank(Image.alpha_composite(self.img, other.img))
        else:
            return Blank(Image.alpha_composite(self.img, other))
    
    def _radd__(self,other):
        if type(other) == {Glow, Blank}:
            return Blank(Image.alpha_composite(other.img, self.img))
        else:
            return Blank(Image.alpha_composite(other, self.img))

    def save(self, filename):
        self.img.save(filename)

    def __mul__(self,other):
        # Convert the image to a NumPy array
        arr = np.array(self.img, dtype=float)

        # Multiply the alpha channel (4th channel, index 3)
        alpha = arr[:, :, 3] * other        
        arr[:, :, 3] = np.clip(alpha, 0, 255)

        return Blank(Image.fromarray(arr.astype(np.uint8), 'RGBA'))


class Glow(Blank):
    def __init__(self, image: np.array, strength:float = 2, radius = 80, glow_color:tuple = None):
        """image can be a PIL image or a RGBA/BGRA numpy array"""
        self.image = np.array(image)
        self.glow_color = glow_color
        self.strength = strength
        self.h, self.w, _ = image.shape
        self.blur = int((radius * min(self.w, self.h))/1000)
        self.img = None
        self.mask = self.image[..., 3]

        self._render()

    def _render(self):
        w, h = self.w, self.h
        bgra_image = np.empty((h, w, 4), dtype=np.uint8)
        bgra_image[..., :3] = self.glow_color[::-1]  # B, G, R
        bgra_image[..., 3] = self.mask


        pil_image = Image.fromarray(bgra_image[..., [2, 1, 0, 3]], 'RGBA')

        glow_radius = self.blur

        outer_glow = pil_image.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        inner_glow = pil_image.filter(ImageFilter.GaussianBlur(radius=glow_radius / 3))

        #These are our layers for the glow
        layers = []
        layers.extend([inner_glow for _ in range(self.strength)])
        layers.extend([outer_glow for _ in range(1 + int(self.strength/3))])
        
        self.img = layers[0]
        for i in range(1,len(layers)):
            self.img = Image.alpha_composite(self.img, layers[i])



def apply_watermark(frame_raw, wm_frame, alpha_mult=1.0, wr=0.3, hr=0.3, x_pos=1.0, y_pos=1.0):
    """
    Applies a BGRA watermark to a BGRA frame with correct alpha blending.
    """
    # Load watermark as BGRA
    wm = cv2.cvtColor(wm_frame, cv2.COLOR_BGR2BGRA)
    frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2BGRA)
    
    wm_bgr = wm[:, :, :3].astype(np.float32)
    wm_alpha = wm[:, :, 3].astype(np.float32) / 255.0

    # Scale alpha
    wm_alpha = np.clip(wm_alpha * alpha_mult, 0, 1)


    fh, fw = frame.shape[:2]
    max_w, max_h = fw * wr, fh * hr


    h, w = wm_alpha.shape
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    wm_bgr = cv2.resize(wm_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    wm_alpha = cv2.resize(wm_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute position
    max_x, max_y = fw - new_w, fh - new_h
    x = int((x_pos + 1) / 2 * max_x)
    y = int((y_pos + 1) / 2 * max_y)



    # Define region of interest (ROI) from the BGRA frame
    roi = frame[y:y + new_h, x:x + new_w]
    
    # Convert ROI to float for calculations
    roi_float = roi.astype(np.float32)
    roi_bgr = roi_float[:, :, :3]
    roi_alpha = roi_float[:, :, 3:] / 255.0 


    wm_alpha_3d = wm_alpha[:, :, None]


    # Calculate the output alpha channel
    out_alpha = wm_alpha_3d + roi_alpha * (1 - wm_alpha_3d)


    blended_bgr_num = (wm_bgr * wm_alpha_3d) + (roi_bgr * roi_alpha * (1 - wm_alpha_3d))

    # Handle division by zero for fully transparent areas
    # Create a mask where the output alpha is not zero
    mask = out_alpha > 1e-6
    blended_bgr = np.zeros_like(roi_bgr)

    np.divide(blended_bgr_num, out_alpha, out=blended_bgr, where=mask)
    
    # Combine the blended BGR and the new alpha channel (scaled back to 0-255)
    blended = np.concatenate((blended_bgr, out_alpha * 255.0), axis=2)
    

    output = frame.copy()
    output[y:y + new_h, x:x + new_w] = np.clip(blended, 0, 255).astype(np.uint8)

    return output

watermark_cache = (None, None)
def get_glowy_watermark_slow(wm_frame, mask1_frame, mask2_frame, strength = 1, glow_color1 = (255, 45, 45), glow_color2 = (219, 104, 41), alpha_multiplier = 1, alpha_multiplier2 = 1):
    global watermark_cache
    if (alpha_multiplier, alpha_multiplier2) == watermark_cache[0]:
        return watermark_cache[1]    
    
    size = (512,512)

    #masks
    mask1 = cv2.resize(mask1_frame, size) 
    mask2 = cv2.resize(mask2_frame, size)
    
    #watermark
    watermark = cv2.resize(wm_frame, size)

    #Shadow layer
    out = Glow(mask1, strength*2, 30, (40,40,40)) * alpha_multiplier
    out += Glow(mask1, strength*2, 250, (40,40,40)) * alpha_multiplier

    #Glow layers
    radii = [30, 200, 80, None, 200] #Radii of layers, None -> actual watermark to be pasted
    for radius in radii:
        if radius is None:
            out += Blank(Image.fromarray(watermark[..., [2, 1, 0, 3]])) * alpha_multiplier2 #Converts to PIL object
            continue
        for mask,glow_color in [(mask1, glow_color1),(mask2,glow_color2)][::-1]:
            out += Glow(mask, strength, radius, glow_color) * alpha_multiplier
        
    out3 = cv2.cvtColor(np.array(out.img), cv2.COLOR_RGBA2BGRA)  
    watermark_cache = ((alpha_multiplier, alpha_multiplier2),out3)
    return out3

watermark_cache_optimized = dict()
def get_glowy_watermark(wm_frame, mask1_frame, mask2_frame, strength = 1, glow_color1 = (255, 45, 45), glow_color2 = (219, 104, 41), alpha_multiplier = 1, alpha_multiplier2 = 1, prog = 0):
    global watermark_cache_optimized
    quantization = 25 #60 unique at most
    key = int(prog * quantization)

    if key in watermark_cache_optimized:
        #print("\n\nCACHED!!\n\n")
        return watermark_cache_optimized[key]
    
    watermark_cache_optimized[key] = get_glowy_watermark_slow(wm_frame, mask1_frame, mask2_frame, strength, glow_color1, glow_color2, alpha_multiplier, alpha_multiplier2)

    return watermark_cache_optimized[key]


def get_glowy_watermark2(wm_frame, mask1_frame, strength = 1, glow_color1 = (255, 45, 45), alpha_multiplier = 1):
    size = (512,512)

    #masks
    mask1 = cv2.resize(mask1_frame, size) 
    
    #watermark
    watermark = cv2.resize(wm_frame, size)

    #Shadow layer
    out = Glow(mask1, strength*2, 30, (40,40,40)) * alpha_multiplier
    out += Glow(mask1, strength*2, 250, (40,40,40)) * alpha_multiplier

    #Glow layers
    radii = [30, 200, 80, None, 200] #Radii of layers, None -> actual watermark to be pasted
    for radius in radii:
        if radius is None:
            out += Image.fromarray(watermark[..., [2, 1, 0, 3]]) #Converts to PIL object
            continue
        for mask,glow_color in [(mask1, glow_color1)][::-1]:
            out += Glow(mask, strength, radius, glow_color) * alpha_multiplier
        
    return cv2.cvtColor(np.array(out.img), cv2.COLOR_RGBA2BGRA)  
    out.save('output.png')

def watermark_applier_adv(frame, mask1, mask2, watermark, alpha1 = 1, alpha2 = 1, prog = 0):
    SIZE = 0.11
    H = 0.9
    frame0 = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    watermark2 = apply_watermark(frame0, watermark, alpha_mult=1.0, wr=SIZE, hr=SIZE, x_pos=0, y_pos=H)
    mask11 = apply_watermark(frame0, mask1, alpha_mult=1.0, wr=SIZE, hr=SIZE, x_pos=0, y_pos=H)
    mask22 = apply_watermark(frame0, mask2, alpha_mult=1.0, wr=SIZE, hr=SIZE, x_pos=0, y_pos=H)
    
    frame_with_wm = get_glowy_watermark(watermark2, mask11, mask22, 3, alpha_multiplier = alpha1, alpha_multiplier2 = alpha2, glow_color1=(255,255,255), glow_color2=(255,255,255), prog = prog)
    return cv2.cvtColor(apply_watermark(frame, frame_with_wm, 1, 1, 1, 0,0), cv2.COLOR_BGRA2BGR)

def watermark_applier(frame, mask1, mask2, watermark, prog_raw):
    MAX_ALPHA = 0.75
    if prog_raw == 0:
        return frame

    if prog_raw < 0.5:
        prog = prog_raw * 2
        return watermark_applier_adv(frame, mask1, mask2, watermark, alpha1 = MAX_ALPHA *  prog, alpha2 = MAX_ALPHA * (0.25*prog), prog = prog_raw)
    else:
        prog = (prog_raw - 0.5)*2
        return watermark_applier_adv(frame, mask1, mask2, watermark, alpha1 = MAX_ALPHA * 1, alpha2 = MAX_ALPHA * (0.75*prog + 0.25), prog = prog_raw)

watermark, mask1, mask2 = None, None, None

def set_up_filter(watermark_path, mask1_path, mask2_path):
    global watermark, mask1, mask2

    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_UNCHANGED)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_UNCHANGED)

current_iter = 0
def main_filter(frame):
    time_steps = [2.5, 0.4, 2.5, 0.4] #No logo, load in, logo, load out
    global watermark, mask1, mask2, current_iter
    
    time_steps = [int(x*FPS) for x in time_steps]
    step = current_iter%sum(time_steps)
    if step < time_steps[0]:
        prog = 0
    elif step < time_steps[0] + time_steps[1]:
        prog = (step - time_steps[0])/time_steps[1]
    elif step < time_steps[0] + time_steps[1] + time_steps[2]:
        prog = 1
    else:
        prog = (step - (time_steps[0] + time_steps[1] + time_steps[2]))/time_steps[3]
        prog = 1 - prog

    current_iter += 1
    return watermark_applier(frame, mask1, mask2, watermark, prog)

if __name__ == "__main__":
    # Example: create a 1-second animation at 30 fps
    frame = cv2.imread("frame.png")
    watermark_path = "watermark3.png"
    fps = 30
    duration = 1.0
    total_frames = int(fps * duration)

    h, w = frame.shape[:2]
    out = cv2.VideoWriter("watermark_animation3.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    mask1 = cv2.imread('mask1.png', cv2.IMREAD_UNCHANGED)
    mask2 = cv2.imread('mask2.png', cv2.IMREAD_UNCHANGED)

    from tqdm import tqdm


    for i in tqdm(range(45)):
        prog = min(i/15, 1)
        out.write(watermark_applier(frame, mask1, mask2, watermark, prog))        
        
    out.release()
    #print("Saved animation: watermark_animation.mp4")
    cv2.imwrite('output.png', watermark_applier_adv(frame, mask1, mask2, watermark, alpha1 = 1, alpha2 = 1))
    cv2.imwrite('output2.png', watermark_applier(frame, mask1, mask2, watermark, 1))
    

