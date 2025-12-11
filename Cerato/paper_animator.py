from caption import Text, PlainText, Glow
import cv2
import numpy as np
import tqdm
import os
import random
from PIL import Image
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
from moviepy.audio.fx import AudioLoop
from settings import VIDEO_SIZE

def get_bbox(data):
    alpha = data[:, :, 3]
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        bbox = None  # no opaque region
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox = (x_min, y_min, x_max, y_max)  # (left, top, right, bottom)

    return bbox


def create_img(text, key_word, output_file, font = "fonts/Caveat.ttf", output_dimensions = (1080,1920), font_size=40, font_color = (33,33,33), glow_color = (255, 36, 36), img_path = None):
    text = text.replace("\n"," ")
    LINE_SPACE = 0.00015
    SPACE_SIZE = 0.0008 * min(*output_dimensions) * font_size
    
    textObj = Text(key_word,w=output_dimensions[0],h=output_dimensions[1],font_path=font, font_size=font_size, pad_x=0,pad_y=0,font_color=font_color,line_space=LINE_SPACE)
    key_word_norm = np.array(PlainText(textObj))
    kwbbbox = get_bbox(key_word_norm)
    lw,tw,rw,bw = kwbbbox
    scale =  0.0318/(((bw - tw) * (rw - lw))/(output_dimensions[0] * output_dimensions[1]))
    zoom_scale = scale**0.5

    if not kwbbbox:
        raise Exception("PROBLEM")
    
    text1, text2 = text.split(key_word) #Error here means not exactly 1 keyword

    text1 = [x for x in text1.split(" ") if len(x) >= 1][::-1] #Reverse since we are adding reversed
    text2 = [x for x in text2.split(" ") if len(x) >= 1]

    ## For debug
    text1 = text1[:min(len(text1), 65)]
    text2 = text2[:min(len(text2), 65)]
    
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (output_dimensions[0]*2, output_dimensions[1]*2), "white")
    draw = ImageDraw.Draw(img)
    fontObj = ImageFont.truetype(font, int(font_size * min(*output_dimensions) * (1/480)))
    ascent, descent = fontObj.getmetrics()
    anchor = (output_dimensions[0],output_dimensions[1])
    bottomost = output_dimensions[1] + ascent + descent
    
    lw,tw,rw,bw = draw.textbbox(anchor, key_word, font=fontObj)
    del_key = (anchor[0] - tw, bottomost - bw)


    text1bbox2 = [draw.textbbox(anchor, word, font=fontObj) for word in text1]
    text2bbox2 = [draw.textbbox(anchor, word, font=fontObj) for word in text2]
    
    text1bbox2 = [(anchor[0] - tw, bottomost - bw) for lw,tw,rw,bw in text1bbox2]
    text2bbox2 = [(anchor[0] - tw, bottomost - bw) for lw,tw,rw,bw in text2bbox2]

    final_result = Image.fromarray(cv2.cvtColor(key_word_norm, cv2.COLOR_BGRA2RGBA)) #PIL_image

    text1 = [Text(word,w=output_dimensions[0],h=output_dimensions[1],font_path=font, font_size=font_size, pad_x=0,pad_y=0,font_color=font_color,line_space=LINE_SPACE) for word in text1]
    text2 = [Text(word,w=output_dimensions[0],h=output_dimensions[1],font_path=font, font_size=font_size, pad_x=0,pad_y=0,font_color=font_color,line_space=LINE_SPACE) for word in text2]

    text1 = [np.array(PlainText(word)) for word in text1]
    text2 = [np.array(PlainText(word)) for word in text2]

    text1bbox = [get_bbox(word_norm) for word_norm in text1]
    text2bbox = [get_bbox(word_norm) for word_norm in text2]
    #max_height = max([bw - tw for lw,tw,rw,bw in text1bbox + text2bbox])    

    #text1bbox = [(lw,(bw+tw)//2 - max_height//2, rw, (bw+tw)//2 + max_height//2) for lw,tw,rw,bw in text1bbox]
    #text2bbox = [(lw,(bw+tw)//2 - max_height//2, rw, (bw+tw)//2 + max_height//2) for lw,tw,rw,bw in text2bbox]

        

    text1bbox = [(lw,tw + text1bbox2[i][0], rw, bw + text1bbox2[i][1]) for i,(lw,tw,rw,bw) in enumerate(text1bbox)]
    text2bbox = [(lw,tw + text2bbox2[i][0], rw, bw + text2bbox2[i][1]) for i,(lw,tw,rw,bw) in enumerate(text2bbox)]
    

    text1 = [Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)).crop(text1bbox[i]) for i,overlay in enumerate(text1)]
    text2 = [Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)).crop(text2bbox[i]) for i,overlay in enumerate(text2)]
    
    max_height = ascent + descent
    
    current_prev = (kwbbbox[0], kwbbbox[3] - max_height + del_key[1])
    flag = False
    for i,word_pil in enumerate(text1):
        lw,_,rw,_ = text1bbox[i]

        if current_prev[0] < 0:
            if current_prev[1]<0:
                if flag:
                    break
                else:
                    flag = True #Do one more iter

            current_prev = (int(output_dimensions[0]*1.1), current_prev[1] - max_height - LINE_SPACE)

        new_paste_pos = (
            int(current_prev[0] - SPACE_SIZE - (rw - lw)),
            int(current_prev[1])
        )
        current_prev = (current_prev[0] - SPACE_SIZE - (rw - lw),current_prev[1])

        final_result.paste(word_pil, new_paste_pos, word_pil)


    current_prev = (int(kwbbbox[2] + SPACE_SIZE), int(kwbbbox[3] - max_height + del_key[1]))
    flag = False
    for i,word_pil in enumerate(text2):
        lw,_,rw,_ = text2bbox[i]

        if current_prev[0] > output_dimensions[0]:
            if current_prev[1] > output_dimensions[1]:
                if flag:
                    break
                else:
                    flag = True #Do one more iter

            current_prev = (-int(output_dimensions[0]*0.1), int(current_prev[1] + max_height + LINE_SPACE))

        final_result.paste(word_pil, current_prev, word_pil)
        
        current_prev = (int(current_prev[0] + SPACE_SIZE + (rw - lw)),current_prev[1])   


    gls = [
        Glow(textObj, 1, 80, glow_color), 
        Glow(textObj, 1, 200, glow_color), 
        Glow(textObj, 1, 30, glow_color),            
    ]

    for i in range(len(gls)):
        gls[i]._render()

    out = gls[1]
    out += PlainText(textObj)
    out += gls[0]
    out += gls[0]
    out += gls[1]
    out += gls[1]
    out += gls[2]
    out += gls[2]

    img2 = Image.fromarray(cv2.cvtColor(np.array(out), cv2.COLOR_BGRA2RGBA))
    final_result = final_result.convert("RGBA")

    img3 = Image.open(img_path).convert("RGBA")
    output_width, output_height = output_dimensions


    width,height = img3.size
    scale = max(output_width/width, output_height/height)
    if scale > 1:
        img3 = img3.resize((int(width*scale) + 1, int(height*scale) + 1))
        width,height = img3.size


    start_x, start_y = random.randint(0, width - output_width), random.randint(0, height - output_height)
    end_x, end_y = start_x + output_width, start_y + output_height

    img3 = img3.crop([start_x, start_y, end_x, end_y])


    def zoom_center(img, zoom_factor=1.2):
        """
        Zoom in/out on a PIL image, keeping output same size, centered.
        Supports zoom_factor < 1 by padding with transparency.
        
        :param img: PIL.Image object
        :param zoom_factor: float, >1 zooms in, <1 zooms out
        :return: PIL.Image object
        """
        w, h = img.size
        
        # Calculate new size
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        # Resize image
        img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
        
        if zoom_factor >= 1:
            # Crop center for zoom in
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            right = left + w
            bottom = top + h
            return img_resized.crop((left, top, right, bottom))
        else:
            # Paste smaller image onto transparent canvas for zoom out
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # transparent
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            canvas.paste(img_resized, (left, top))
            return canvas


    overlayed = Image.alpha_composite(final_result, img2)

    overlayed = zoom_center(overlayed, zoom_scale)

    overlayed = Image.alpha_composite(img3, overlayed)

    overlayed.save(output_file)

    return True


def create_paper_video(key_word, text_raw, output_file, output_dimensions = (720,720), temp_folder='temp'):
    text_raw = text_raw.replace("\n", " ").replace("  ", " ")
    font_folder = "assets/paper/fonts"
    paper_folder = folder = "assets/paper/paper"
    audio_file = "assets/paper/shutter.wav"
    font_size = (45,45)

    number_of_frames = 16
    time_per_frame = 0.08

    length_of_video = number_of_frames * time_per_frame #In seconds
    image_duration = length_of_video/number_of_frames
    
    efficiency_mask = 35 #Maximum words = efficiency_mask*2 + 1, used to reduce render time. Increase if there isnt text above and below

    text_raw = text_raw.replace(key_word.lower(), key_word)
    text = text_raw.split(" ")
    indices = [i for i,x in enumerate(text) if x == key_word]


    frames = []
    
    files = [f for f in os.listdir(font_folder) if f.lower().endswith(".ttf")]
    files2 = [f for f in os.listdir(folder) if f.lower().endswith(".png")]

    os.makedirs(temp_folder, exist_ok=True)
    output_names = []
    prev_frame = [(None, None, None)]

    pbar = tqdm.tqdm(total=number_of_frames)

    for full_ind in range(100):
        still_left = []

        for _ in range(number_of_frames):
            font = os.path.join(font_folder, random.choice(files))
            img_path = os.path.join(folder, random.choice(files2))
            indice = random.choice(indices)

            if (font, img_path, indice) not in frames:
                frames.append((font, img_path, indice))  
                still_left.append((font, img_path, indice)) 
            
            img_path = os.path.join(folder, random.choice(files2))
            indice = random.choice(indices)
            if (font, img_path, indice) not in frames:
                frames.append((font, img_path, indice))  
                still_left.append((font, img_path, indice)) 


        for i,pkg in tqdm.tqdm(enumerate(still_left)):
            if pkg[2] == prev_frame[-1][2]:
                frames.remove(pkg) #Allow it to happen again later
                continue
            
            font, img_path, indice = pkg
            
            #dist = min(indice, len(text) - indice, efficiency_mask)
            #my_text = " ".join(text[indice - dist: indice + dist])
            #my_ind = dist

            new_indice = len(" ".join(text[:indice])) + len(key_word)//2
            dist = efficiency_mask * 5
            ind1, ind2 = new_indice - dist, new_indice + dist

            my_text = " ".join(text)
            my_text = my_text[max(ind1,0):min(ind2, len(my_text))]
            my_ind = [i for i,x in enumerate(my_text.split(" ")) if x == key_word]
            my_ind = my_ind[len(my_ind)//2]


            out_frame_path = os.path.join(temp_folder, f"paper{i}_{full_ind}.png")
            
            worked = create_img(my_text, key_word, out_frame_path, font=font, output_dimensions=output_dimensions,img_path=img_path,font_size=random.randint(*font_size))
            if worked:
                #print(f"\nCreated frame {out_frame_path} with {font}, {indice}, {img_path}\n")
                output_names.append(out_frame_path)
                prev_frame.append(pkg)
                pbar.update(1)
            
            if len(output_names) >= number_of_frames:
                break
        
        if len(output_names) >= number_of_frames:
            break


    image_clips = [ImageClip(img).with_duration(image_duration) for img in output_names]
    final_video = concatenate_videoclips(image_clips)
    audio_clip = AudioFileClip(audio_file).with_effects([AudioLoop(duration=final_video.duration)])
    
    final_video = final_video.with_audio(audio_clip)
    final_video.write_videofile(output_file, fps=60, codec='libx264')  

   

text_block = """The Python-Powered Video Editor Redefining Automated Creativity
In a world dominated by industry giants like Adobe After Effects, a new contender has emerged from the open-source community, offering a radically different approach to video editing and motion graphics. Cerato, a completely free and open-source video editing software, is built on a foundation of Python, promising a level of flexibility, stability, and intelligent automation that aims to surpass the capabilities of its well-established rivals. With a unique focus on a minimalist interface and powerful, music-driven editing, Cerato is carving out a niche for a new generation of creators who value efficiency and customizability.
A Foundation of Code: The Power of Python and Open Source
At its core, Cerato has the defining characteristic of being fully Python-based. This architectural choice is a significant departure from the proprietary, closed systems of most professional editing suites. By leveraging Python, a versatile and widely-used programming language, Cerato offers users unprecedented access to its inner workings.[1][2][3] This allows for easy additions and modifications, empowering developers and technically-minded artists to build custom tools, automate repetitive tasks, and integrate Cerato into larger production pipelines with ease.
Being open-source and free, Cerato removes the significant financial barrier to entry that often comes with professional-grade software.[4][5] This philosophy fosters a collaborative environment where users can contribute to the software's development, ensuring it evolves based on the community's needs.
The Rhythm of the Edit: Intelligent Beat-Synced Transitions
Perhaps the most  groundbreaking feature in Cerato is its advanced beat detection system. While many editors offer tools to identify transients in an audio track, Cerato takes this a step further by using this data to make intelligent creative decisions.[6][7][8] The software automatically analyzes a song and its media, and then selects and applies transitions that best match the rhythm and mood of the piece.
What makes this system particularly powerful is its library of versatile, hand-coded transitions. Rather than relying on a standard set of wipes and dissolves, Cerato has intelligent algorithm pulls from this unique collection to create dynamic, musically-synchronized sequences automatically. This feature can dramatically accelerate the editing process for music videos, promotional content, and any project where the soundtrack is a driving force.
Advanced Color Correction and the "LUT Replicator"
Cerato comes equipped with a robust color correction toolset. It includes native tools for granular adjustments and supports the use of Look-Up Tables (LUTs) for quick and consistent color grading. However, its standout feature is the innovative 'LUT Replicator Tool'. This powerful function allows users to analyze a reference image or video clip and generate a LUT that replicates its color profile. This can be an invaluable tool for matching footage from different cameras or emulating the aesthetic of a favorite film, a process that often requires highly specialized and expensive software.[9]
A Minimalist Ethos: A Streamlined and Stable Workflow
In an era of increasingly complex user interfaces, Cerato champions a minimalist, terminal-based approach. This design choice strips away the visual clutter of traditional non-linear editors, presenting a "super linear" workflow that is focused and efficient. While this may be a departure for those accustomed to drag-and-drop interfaces, it offers a highly streamlined and distraction-free environment for creators who are comfortable with a command-line-driven process.
This simplicity contributes to what is described as one of its greatest assets - Cerato is extremely stability. By minimizing interface overhead and focusing on a clean, code-based foundation, the software is designed to handle projects of any resolution without the frequent crashes that can plague more complex applications.
Project management is handled with elegant simplicity. All project files, media, and editable .json files for nuanced adjustments are neatly contained within a single "Cerato" folder chosen by the user, ensuring projects are portable and self-contained.
Who is Cerato For?
Cerato is not positioned as a replacement for every editor on the market. Its unique, code-centric approach makes it the ideal tool for a specific type of user:
Developers and Technical Artists: Those who can leverage the Python backbone to create custom tools and workflows.
Music-Driven Content Creators: Editors who want to harness the power of automated, beat-synced transitions to produce dynamic content quickly.
Minimalists and Efficiency Enthusiasts: Creators who prefer a streamlined, terminal-based workflow free from the clutter of traditional GUIs.
The Open-Source Community: Anyone who believes in the power of free, collaborative software development.
By combining the logical power of Python with intelligent creative automation, Cerato presents a compelling vision for the future of video editing—one that is more customizable, accessible, and intelligently streamlined than ever before.
""".replace("\n", " ").replace("  "," ")

# Test Block :D
#create_paper_video("Cerato", text_block, output_file='test.mp4', temp_folder='temp')

def text_generator(keyword, adjectives):
    #Clean up
    adjectives = adjectives.replace("-", " ").replace(",", " ").replace("—", " ")

    adjectives = "".join([char for char in adjectives if char.isalpha() or char == " "])
    adjectives = adjectives.replace(keyword, "").replace(keyword.lower(), "").replace("  ", " ").replace("  ", " ")

    adjectives = adjectives.split(" ")
    adjectives = [x for x in adjectives if len(x) >= 3]
    
    if len(adjectives) < 200:
        raise Exception("Not enough adjectives.")
    
    
    text = ""      
    for _ in range(100):
        random.shuffle(adjectives)

        text += " ".join(adjectives[:50]) + f" {keyword} " + " ".join(adjectives[-50:])

    return text


def launch(filepath = None, temp_folder = 'temp'):
    from tkinter import filedialog
    import ctypes
    try: 
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    keyword = input("Enter in keyword: ")

    print("You now must create a file with 200 related words that will be flashed as needed...")
    input("Click enter to continue...")
    adj_path = filedialog.askopenfilename(
        title="Select your txt file with adjectives in it (atleast 200)",
        filetypes=[("All Files", "*.*")]    
    )
    print(adj_path)
    
    with open(adj_path) as f:
        text_block = text_generator(keyword, f.read().strip())
    
    save_path = filepath if filepath is not None else filedialog.asksaveasfilename(
        title="Select your where to save the output",
        filetypes=[("Video files", "*.mp4")] 
    )   
    print(save_path)
    print("this will take a while...")
    create_paper_video(keyword, text_block, output_file=save_path, output_dimensions=VIDEO_SIZE, temp_folder=temp_folder)

if __name__ == "__main__":
    launch()