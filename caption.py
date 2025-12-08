import whisper_timestamped as whisper
import torch
import subprocess
import time
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from PIL import Image, ImageFilter
import copy
import tqdm
import settings

def create_caption_json_ai(filename, output_filename, max_iters = 3, error = None):
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)  # give it a moment to start

    import ollama
    if max_iters <= 0:
        raise Exception(f"Max iterations tried. Unable to construct captions. Err: {error}")
    
    audio = whisper.load_audio(filename)
    model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    result = whisper.transcribe(model, audio, language="en")

    word_timestamps = []
    for segment in result["segments"]:
        for word in segment["words"]:
            word_timestamps.append((word["text"], word["start"], word["end"]))


    # system instruction and the user prompt

    with open("prompt.txt") as f:
        system_instruction = f.read()

    system_instruction = system_instruction.split("$")
    font_names = system_instruction[1].split("\n")
    a,b = font_names[:1], font_names[1:]
    random.shuffle(b)
    font_names = a + b
    system_instruction = system_instruction[0] + "\n".join(font_names) + system_instruction[2]
    
    
    user_prompt = " ".join([i for i,_,_ in word_timestamps]).replace("\n", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
    
    # The messages payload
    messages = [
        {
            'role': 'system',
            'content': system_instruction,
        },
        {
            'role': 'user',
            'content': user_prompt,
        },
    ]

    
    response_chunks = []
    
    stream = ollama.chat(
        model='llama3.1',
        messages=messages,
        stream=True,
    )

    print(f"Transcript: {user_prompt}")
    print("LLM output >>> ", end="", flush=True)

    # Iterates over the stream to print each chunk and save it
    for chunk in stream:
        content = chunk['message']['content']        
        print(content, end='', flush=True)
        response_chunks.append(content)

    
    full_response = "".join(response_chunks)

    
    print("\n--- Full Response ---")
    print(full_response)
    print("---------------------------------------")


    out2 = word_timestamps
    out = full_response


    out2 = [(a, float(b), float(c)) for a,b,c in out2] #Castes types


    past_open = None
    past_close = None
    for i,char in enumerate(out):
        if char == '[':
            past_open = int(i)
        elif char == ']' and past_open is not None:
            past_close = int(i)

    if past_open is None or past_close is None:
        create_caption_json_ai(filename, output_filename, max_iters-1 ,"LLM output has no parsable text")

    out = out[past_open: past_close + 1]

    data = json.loads(out)
    if not isinstance(data, (dict, list)):
        create_caption_json_ai(filename, output_filename, max_iters-1 ,"JSON must be an object or array")


    for i in range(len(data)):
        data[i]['text'] = data[i]['text'].split(" ")


    data2 = []
    for i in range(len(data)):
        for j in range(len(data[i]['text'])):
            data2.append({'text': data[i]['text'][j],
                'font': data[i]['font'],
                'color': data[i]['color'], 
                'glow': data[i]['glow'], 
                'key': data[i]['key'] if len(data[i]['text']) == 1 else False, 
                'glowboost': data[i]['glowboost']
            })


    def is_equal(str1, str2):
        return "".join([char for char in str1 if char.isalpha()]).lower().strip() == "".join([char for char in str2 if char.isalpha()]).lower().strip()

    words1 = [a for a,b,c in out2]
    words2 = [a['text'] for a in data2]

    valid = all([is_equal(words1[i], words2[i]) for i in range(min(len(words1), len(words2)))])

    if valid and len(words1) == len(words2):
        print("LLM output is valid. Proceeding with text creation")
    else:
        return create_caption_json_ai(filename, output_filename, max_iters-1 ,"LLM output invalid. Does not match transcript.")

    def pseudo_strip(str1):
        return "".join([char for char in str1 if char.isalpha()]).strip()
        

    for i in range(len(data2)):
        data2[i]['start'] = [out2[i][1]]
        data2[i]['end'] = [out2[i][2]]
        data2[i]['text'] = pseudo_strip(data2[i]['text'])


    max_combine_words = 4
    def compatible(d1, d2):
        if d1 is None or d2 is None:
            return False
        if d1['key'] or d2['key']:
            return False
        for key in d1.keys():
            if not key in {'text', 'start', 'end'}:
                if d1[key] != d2[key]:
                    return False
        return True

    def print_dict(d1):
        print("\n".join([str(x) for x in d1]))


    data3 = [data2[0]]

    for item in data2[1:]:
        if compatible(data3[-1], item) and len(data3[-1]['text'].split(" ")) < max_combine_words:
            data3[-1]['text'] += " " + item['text']
            data3[-1]['start'].extend(item['start'])
            data3[-1]['end'].extend(item['end'])
            
        else:
            data3.append(item)

    print("\n\n\n\n")


    print_dict(data3)

    with open(output_filename, 'w') as f:
        f.write('[\n')
        for item in data3[:-1]:
            json.dump(item, f)
            f.write(',\n')
        
        json.dump(data3[-1], f)
        f.write('\n]')


def create_caption_json(filename, output_filename, max_iters = 3, error = None):
    if max_iters <= 0:
        raise Exception(f"Max iterations tried. Unable to construct captions. Err: {error}")
    
    audio = whisper.load_audio(filename)
    model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    result = whisper.transcribe(model, audio, language="en")

    word_timestamps = []
    for segment in result["segments"]:
        for word in segment["words"]:
            word_timestamps.append((word["text"], word["start"], word["end"]))




    out2 = word_timestamps
    out2 = [(a, float(b), float(c)) for a,b,c in out2]

    def get_color():
        #For reference:
        #colors = [(0,255,255), (255, 105, 180), (176, 38, 255), (255, 165, 0), (255, 36, 36)] #Blue, Purple, Pink, Orange, Red
        colors = ['#00FFFF', '#FF69B4', '#B026FF', '#FFA500', '#FF2424']

        chance_for_colored = 0.38

        for _ in range(0,1_000_000):
            col = random.choice(colors) if random.random() <= chance_for_colored else "#FF2424" #This is the default glow
            for _ in range(4):
                yield "#FFFFFF" #Text color
                yield col #Glow color
                
    def get_font():
        fonts = ["Cinzel", "Merriweather"]
        chance = 0.25

        for _ in range(0,1_000_000):
            col = random.choice(fonts) if random.random() <= chance else "Default"
            for _ in range(4):
                yield col


    col_get = get_color()
    font_get = get_font()

    data = [{"text": word.upper(), "font": next(font_get), "color": next(col_get), "glow": next(col_get), "key": False, "glowboost": False} for word,_,_ in word_timestamps]

    for i in range(len(data)):
        data[i]['text'] = data[i]['text'].split(" ")


    data2 = []
    for i in range(len(data)):
        for j in range(len(data[i]['text'])):
            data2.append({'text': data[i]['text'][j],
                'font': data[i]['font'],
                'color': data[i]['color'], 
                'glow': data[i]['glow'], 
                'key': data[i]['key'] if len(data[i]['text']) == 1 else False, 
                'glowboost': data[i]['glowboost']
            })


    def is_equal(str1, str2):
        return "".join([char for char in str1 if char.isalpha()]).lower().strip() == "".join([char for char in str2 if char.isalpha()]).lower().strip()

    words1 = [a for a,b,c in out2]
    words2 = [a['text'] for a in data2]

    valid = all([is_equal(words1[i], words2[i]) for i in range(min(len(words1), len(words2)))])

    if valid and len(words1) == len(words2):
        pass
    else:
        return create_caption_json(filename, output_filename, max_iters-1 ,"LLM output invalid. Does not match transcript.")

    def pseudo_strip(str1):
        return "".join([char for char in str1 if char.isalpha()]).strip()
        

    for i in range(len(data2)):
        data2[i]['start'] = [out2[i][1]]
        data2[i]['end'] = [out2[i][2]]
        data2[i]['text'] = pseudo_strip(data2[i]['text'])


    max_combine_words = 4
    max_dist = 1
    def compatible(d1, d2):
        if d1 is None or d2 is None:
            return False
        if d1['key'] or d2['key']:
            return False
        for key in d1.keys():
            if not key in {'text', 'start', 'end'}:
                if d1[key] != d2[key]:
                    return False

        if d2['start'][0] - d1['end'][-1] > max_dist: #Words are too far away
            return False

        return True

    def print_dict(d1):
        print("\n".join([str(x) for x in d1]))

    if len(data2) == 0:
        with open(output_filename, 'w') as f:
            f.write("[]")
        return

    data3 = [data2[0]]

    for item in data2[1:]:
        if compatible(data3[-1], item) and len(data3[-1]['text'].split(" ")) < max_combine_words:
            data3[-1]['text'] += " " + item['text']
            data3[-1]['end'][-1] = item['start'][0]
            data3[-1]['start'].extend(item['start'])
            data3[-1]['end'].extend(item['end'])
            
        else:
            data3.append(item)

    print("\n\n\n\n")


    print_dict(data3)

    with open(output_filename, 'w') as f:
        f.write('[\n')
        for item in data3[:-1]:
            json.dump(item, f)
            f.write(',\n')
        
        json.dump(data3[-1], f)
        f.write('\n]')



class Text:
    def __init__(self, text, w, h, font_path, font_size = 60, font_color = (255,255,255), stroke_width=0, outline_width = 0, pad_x = 0.25, pad_y = 0.25, line_space = 0.0003, auto_wrap = True):
        """
        w,h = output layer width -> use the final image's w/h
        self.text = Your ENTIRE text
        self.font_path = .ttf file
        font_size = relative font size (scales automatically with resolution)
        outline_width = outline width of text.
        stroke_width = stroke width of text 0 -> min stroke width
        pad_x = relative padding of width to width [0,1] #Settings this to a negative number can nullify the autowrap, by providing extra space beyond the frame (good for custom classes)
        pad_y = relative padding of height to height [0,1]
        line_space = relative line spacing between adjacent lines
        auto_wrap = True for word based wrapping, False for letter based text wrap. Letter based is ideal for animations requiring single letters only. However do not expect great text - wrapping        
        """
        self.text = text
        self.w = w
        self.h = h
        try:
            self.font = ImageFont.truetype(font_path, int(font_size * min(w,h) * (1/480)))
        except:
            if not os.path.exists(font_path):
                print(f"{font_path} does not exist")
                raise FileNotFoundError
        self.font_color = font_color
        self.outline_width = outline_width
        self.stroke_width = stroke_width
        self.visible = [i for i in range(len(text.split(" ") if auto_wrap else text))]
        self.safe_region = (w * (1-2*pad_x), h * (1-2*pad_y))
        self.line_spacing = h * line_space * font_size
        self.pad_x = pad_x * w
        self.pad_y = pad_y * h
        self.auto_wrap = auto_wrap
    
    def __len__(self):
        if self.auto_wrap:
            return len(self.text.split(" "))
        return len(self.text)

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self,index):
        max_len = len(self)

        if type(index) == int:
            visible = [index]
        if type(index) == slice:
            index = (index, ) #Covnerts to tuple
        if type(index) == tuple:
            visible = []
            for item in index:
                if type(item) == int:
                    visible.append(item)
                if type(item) == slice:
                    start = item.start if item.start is not None else 0
                    stop = item.stop if item.stop is not None else max_len
                    step = item.step if item.step is not None else 1
                    for i in range(start, stop, step):
                        visible.append(i)
            
        for item in visible:
            if item >= max_len or item < 0:
                raise IndexError
            
        new_obj = self.copy()
        new_obj.visible = set(visible)
        
        return new_obj
    
    def __str__(self):
        if self.auto_wrap:
            blocks = self.text.split(" ")
        else:
            blocks = list(self.text)
        
        blocks = [item if i in self.visible else f"*{item}*" for i,item in enumerate(blocks)]

        return f"Text: {(' ' if self.auto_wrap else '').join(blocks)}"


class AccurateDraw:
    def __init__(self, textObj):
        self.textObj = textObj
            
    def textbbox(self, *args, **kwargs):
        overlay_pil = Image.new("RGBA", (self.textObj.w * 3, self.textObj.h * 3), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_pil)

        new_arg1 = (args[0][0] * self.textObj.w, args[0][1] * self.textObj.h) #Ummmm just realised this doesnt handle if the first argument is in kwargs so fix that if issues aaya tho, but it works for now

        draw.text(new_arg1, *args[1:], fill=(255,255,255, 255), **kwargs)
        rgba_array = np.array(overlay_pil)

        alpha_channel = rgba_array[:, :, 3]
        non_zero_coords = np.nonzero(alpha_channel)

        if non_zero_coords[0].size == 0:
            return (0,0,0,0)
        else:
            # Find the bounding box
            min_y, max_y = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
            min_x, max_x = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])

            # Calculate width and height
            bounding_box_width = max_x - min_x + 1
            bounding_box_height = max_y - min_y + 1
            return (0,0,bounding_box_width, bounding_box_height)             

  
class TextLayer:
    def __init__(self, textObj: Text, bgra = None):
        self.bgra = bgra
        self.textObj = textObj

    def copy(self):
        return copy.deepcopy(self)
    
    def _render(self):
        raise Exception("Default render function called")

    def __str__(self):
        return str(self.textObj)

    def __getitem__(self,index):
        new_obj = self.copy()
        new_obj.textObj = self.textObj[index]
        return new_obj

    def __add__(l1, l2): #Here l1 is technically self. but regardless, this is l1 + l2
        if l1.bgra is None:
            l1._render()
        if l2.bgra is None:
            l2._render()

        a_fg = l1.bgra[..., 3:4] / 255 #extra alphas
        a_bg = l2.bgra[..., 3:4] / 255

        # Alpha blend
        out_rgb = l1.bgra[..., :3] * a_fg + l2.bgra[..., :3] * a_bg * (1 - a_fg)
        out_a = a_fg + a_bg * (1 - a_fg)


        return TextLayer(l1.textObj if l1.textObj is not None else l2.textObj, np.concatenate((out_rgb / np.clip(out_a, 1e-6, 1), out_a * 255), axis=-1).astype(np.uint8))

    def __radd__(l2,x): #for x + l2, but x doesnt know how to add
        if type(x) in {int, float}:
            return l2

    def __array__(self, dtype=None, copy = None):
        if self.bgra is None:
            self._render()

        if dtype:
            return np.array(self.bgra, dtype=dtype)
        return self.bgra
    
    def __len__(self):
        return len(self.textObj)
    
    def __mul__(self, other):
        if type(other) in {int, float}:
            if self.bgra is None:
                self._render()

            self.bgra[..., 3] = np.clip(self.bgra[..., 3].astype(np.float32) * other, 0, 255).astype(np.uint8)
            return self
    
    def _rmul_(self, other):
        return self.__mul__(other)

    def create_raw_mask(self):
        if self.textObj.auto_wrap:
            letters = self.textObj.text.split(" ")
            additional_delimeter = " "
        else:
            letters = [letter for letter in self.textObj.text] #Can be shifted to word-based text wrapping by changing letter -> words via letters = self.textObj.text.split(" ")
            additional_delimeter = ""


        draw = AccurateDraw(textObj=self.textObj.copy()) #replaces ImageDraw.Draw(overlay_pil) -> PIL please fix your code ;-;

        _, top, _, bottom = draw.textbbox((0, 0), self.textObj.text , font=self.textObj.font, stroke_width=self.textObj.stroke_width)
        max_height = bottom - top

        lines = [[]]
        current_line = ""
        current_line_pos = self.textObj.pad_y
        for i, letter in enumerate(letters):
            left, _, right, _ = draw.textbbox((0, 0), current_line + additional_delimeter + letter, font=self.textObj.font, stroke_width=self.textObj.stroke_width)
            new_line_width = right - left

            if new_line_width > self.textObj.safe_region[0] and current_line != "": #For potential off-case
                current_line = ""
                current_line_pos += max_height + self.textObj.line_spacing
                new_char_pos = (self.textObj.pad_x,current_line_pos)
                lines.append([])
                lines[-1].append(new_char_pos)
                current_line += letter
            else:
                left, _, right, _ = draw.textbbox((0, 0), letter, font=self.textObj.font, stroke_width=self.textObj.stroke_width)
                new_char_width = right - left
                new_char_pos = (new_line_width - new_char_width + self.textObj.pad_x, current_line_pos)
                lines[-1].append(new_char_pos)
                current_line += additional_delimeter + letter

        
        #now to center align each line
        current_ind = -1
        for i in range(len(lines)):
            current_ind += len(lines[i])
            left, _, right, _ = draw.textbbox((0, 0), letters[current_ind], font=self.textObj.font, stroke_width=self.textObj.stroke_width)
            new_char_width = right - left
            
            extra_pad = self.textObj.pad_x + self.textObj.safe_region[0] - lines[i][-1][0] - new_char_width
            lines[i] = [(pos_x + extra_pad/2, pos_y) for pos_x, pos_y in lines[i]]

        #to vertically align the lines
        min_y = min([min([pos_y for _, pos_y in line]) for line in lines])
        max_y = max([max([pos_y for _, pos_y in line]) for line in lines]) + max_height
        current_avg = (max_y + min_y)/2
        target_avg = self.textObj.h/2
        del_y = target_avg - current_avg - (max_height/2) #Not sure why but somewhere this happened.
        lines = [[(pos_x, pos_y + del_y) for pos_x, pos_y in line] for line in lines]
        

        #flatten lines
        final_positions = []
        for line in lines:
            for pos in line:
                final_positions.append(pos)

        #Finally render out the pil 
        overlay_pil = Image.new("RGBA", (self.textObj.w, self.textObj.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_pil) #Now use PIL's actual thing

        for i, letter in enumerate(letters):
            if i not in self.textObj.visible:
                continue
            draw.text(final_positions[i], letter, font=self.textObj.font, fill=(255,255,255, 255), stroke_width=self.textObj.stroke_width)

        rgba_array = np.array(overlay_pil)
        bgra_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGRA)
        mask = bgra_array[:, :, 3]

        self.mask = mask

class PlainText(TextLayer):
    def __init__(self, textObj: Text):
        self.textObj = textObj
        self.bgra = None

    def _render(self):
        self.create_raw_mask()

        w, h = self.textObj.w, self.textObj.h
        self.bgra = np.empty((h, w, 4), dtype=np.uint8)
        self.bgra[..., :3] = self.textObj.font_color[::-1]  # B, G, R
        self.bgra[..., 3] = self.mask

class Blank(TextLayer):
    def __init__(self, bgra, textObj = None):
        self.bgra = bgra
        self.textObj = None
          
class Glow(TextLayer):
    def __init__(self, textObj: Text, strength:float = 2, radius = 80, glow_color:tuple = None):
        self.textObj = textObj
        self.glow_color = textObj.font_color if glow_color is None else glow_color
        self.strength = strength
        self.blur = radius * (min(self.textObj.w, self.textObj.h)/1000)
        self.bgra = None

    def _render(self):
        self.create_raw_mask()
        w, h = self.textObj.w, self.textObj.h
        bgra_image = np.empty((h, w, 4), dtype=np.uint8)
        bgra_image[..., :3] = self.glow_color[::-1]  # B, G, R
        bgra_image[..., 3] = self.mask

        #Convert to PIL
        pil_image = Image.fromarray(bgra_image[..., [2, 1, 0, 3]], 'RGBA')

        glow_radius = self.blur

        outer_glow = pil_image.filter(ImageFilter.GaussianBlur(radius=glow_radius))

        inner_glow = pil_image.filter(ImageFilter.GaussianBlur(radius=glow_radius / 3))

        #These are our layers for the glow
        layers = []
        layers.extend([inner_glow for _ in range(self.strength)])
        layers.extend([outer_glow for _ in range(1 + int(self.strength/3))])
        

        #Convert to TextLayer objects
        layers = [np.array(x)[..., [2, 1, 0, 3]] for x in layers] #Convert to BGRA
        layers = [Blank(x) for x in layers]

        self.bgra = sum(layers).bgra


def create_captions(input_video, output_video, transcript_path,DEFAULT_FONT = "ReadexPro", FONT_SIZE_VALS = (40,25),additonal_filter = lambda x:x):
    print(f"Creating caption video for {input_video} -> {output_video}")

    def pseudo_strip(str1):
        return "".join([char for char in str1 if char.isalpha()]).strip()

    def create_text_overlay(size:tuple, text:str, font:str, color:str, glow:str, glow_boost:bool, key:bool, till = None,fade = 0):
        FONT_SIZE = FONT_SIZE_VALS[0] if key else FONT_SIZE_VALS[1]

        w,h = size
        glow_color = tuple(int(glow[i:i+2], 16) for i in (1, 3, 5)) #Convert hexadecimal to RGB    
        font_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) #Convert hexadecimal to RGB    

        font = pseudo_strip(font)
        my_text = Text(text, w, h, f"assets/fonts/{font if font != 'Default' else DEFAULT_FONT}.ttf", font_size=FONT_SIZE, font_color=font_color)
        
        if till is not None:
            my_text = my_text[:till]
        
        out = Glow(my_text, 1, 200, glow_color) * (1 if glow_boost else .5) #Overlay-glow-smudge thingy
        out += PlainText(my_text)
        out += Glow(my_text, 1, 80, glow_color)
        out += Glow(my_text, 1, 200, glow_color)  
        out += Glow(my_text, 1, 30, glow_color) 
        if glow_boost:
            out += Glow(my_text, 1, 50, glow_color)
        out += Glow(my_text, 1, 30, (40,40,40)) #Add shadow
        
        return np.array(out)

    if not os.path.exists(transcript_path):
        create_caption_json(input_video, transcript_path, max_iters=3)

    with open(transcript_path) as f:
        data3 = json.load(f)

    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    SIZE = (width, height)
    final_overlays = []
    final_overlay_fade_timings = []

    pbar = tqdm.tqdm(total = sum([len(x['text'].split(" ")) for x in data3]))

    for i in range(len(data3)):
        current = data3[i]['text']
        earliest_start = (data3[i]['start'][0] + (0 if i == 0 else data3[i - 1]['end'][-1]))/2
        latest_end = (data3[i]['end'][-1] + data3[i + 1]['start'][0])/2 if i != len(data3) - 1 else data3[-1]['end'][-1]
        
        allow_fade = True
        for j in range(len(current.split(" "))):
            if not (data3[i]['start'][j] - earliest_start > settings.TEXT_FADE_LEN and latest_end - data3[i]['end'][j]):
                allow_fade = False
                break
        
        for j in range(len(current.split(" "))):
            final_overlays.append((create_text_overlay(
                size=SIZE,
                text = current,
                font = data3[i]['font'],
                color= data3[i]['color'],
                glow = data3[i]['glow'],
                glow_boost=data3[i]['glowboost'],
                key=data3[i]['key'],
                till = j + 1
            ), data3[i]['start'][j], data3[i]['end'][j]))

            """final_overlay_fade_timings.append((
                True if data3[i]['start'][j] - earliest_start > settings.TEXT_FADE_LEN else False,
                True if latest_end - data3[i]['end'][j] > settings.TEXT_FADE_LEN else False                
            ))"""

            """final_overlay_fade_timings.append(
                (True,True) if data3[i]['start'][j] - earliest_start > settings.TEXT_FADE_LEN and latest_end - data3[i]['end'][j] > settings.TEXT_FADE_LEN else (False,False)
            )"""
            final_overlay_fade_timings.append((allow_fade, allow_fade))
            pbar.update(1)
    
    pbar.close()
        


    def composite(base,overlay):
        if overlay == []:
            return base
        
        composited_pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGBA))
        overlay_rgba = [Image.fromarray(cv2.cvtColor(overl, cv2.COLOR_BGRA2RGBA)) for overl in overlay]
        
        for overl_rgba in overlay_rgba:
            composited_pil = Image.alpha_composite(composited_pil, overl_rgba)
        
        return cv2.cvtColor(np.array(composited_pil), cv2.COLOR_RGBA2BGR)


    # Create a VideoCapture object to read the input video
    cap = cv2.VideoCapture(input_video)

    # Get video properties (width, height, and fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp.mp4', fourcc, fps, (frame_width, frame_height))


    final_overlays = [(a,b*fps,c*fps) for a,b,c in final_overlays]
    print([x[1:] for x in final_overlays])

    fade_len = settings.TEXT_FADE_LEN * fps
    fade_prog = settings.TEXT_FADE_PROG

    current_ind = 0
    #while cap.isOpened():
    for _ in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):

        ret, frame = cap.read()

        if ret:
            overlay = []
            for ind,pkg in enumerate(final_overlays):
                overlay2,start,stop = pkg

                if start <= current_ind <= stop and settings.TEXT_ENABLE[1]:
                    overlay.append(overlay2.copy())
                    if settings.TEXT_FAST_SPEAK:
                        break
                
                if start - fade_len < current_ind < start and settings.TEXT_ENABLE[0] and final_overlay_fade_timings[ind][0]:
                    fade_amt = fade_prog((current_ind - (start - fade_len))/fade_len)
                    overlay3 = overlay2.copy()
                    overlay3[..., 3] = (overlay3[..., 3].astype(np.float32) * fade_amt).astype(np.uint8)

                    overlay.append(overlay3.copy())

                    if settings.TEXT_FAST_SPEAK:
                        break
                
                if stop < current_ind < stop + fade_len and settings.TEXT_ENABLE[2] and final_overlay_fade_timings[ind][1]:
                    fade_amt = fade_prog((stop + fade_len - current_ind)/fade_len)
                    overlay3 = overlay2.copy()
                    overlay3[..., 3] = (overlay3[..., 3].astype(np.float32) * fade_amt).astype(np.uint8)

                    overlay.append(overlay3.copy())
                    
                    if settings.TEXT_FAST_SPEAK:
                        break
                
                

            out.write(additonal_filter(composite(frame, overlay))) #Fun little mistake: it should scale the alpha values of the overlays to ensure that each word is not applied multiple times, especially the first words. But it honestly creates this dope enlarging/rescinding spawn-in glow.

        else:
            break
        
        current_ind += 1


    print("Output video saved successfully.")
    cap.release()
    out.release()

    os.system(f'ffmpeg -i temp.mp4 -i "{input_video}" -y -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_video}"')

    #Clean up
    os.remove('temp.mp4')