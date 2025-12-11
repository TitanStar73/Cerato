import cv2
import os
from rembg import remove, new_session
import settings

session = new_session(settings.RB_MODEL)  # 'u2net_human_seg' or 'isnet-general-use'

def rembg_video(video_path, output_folder):
    global session
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from tqdm import tqdm
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"{frame_count}.png")
        cv2.imwrite("temp.png", frame)
        frame_count += 1
        with open("temp.png", "rb") as input_file:
            input_data = input_file.read()

        output_data = remove(input_data, session=session)

        with open(frame_path, "wb") as output_file:
            output_file.write(output_data)

    cap.release()
    print(f"Extracted {frame_count} frames.")
    os.remove("temp.png")
