import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# Import finetuned YOLOv8 model and OCR model
model = YOLO(r"D:\UTKARSH\CV\license_plate_best.pt")  # Replace with your finetuned YOLOv8 model path
reader = easyocr.Reader(['en'],gpu = True)  # Initialize easyocr reader for english

# regex: 2 letters, 2 digits, 3 letters 
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")


mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}
mapping_alpha_to_num = {v: k for k, v in mapping_num_to_alpha.items()}

# mapping to closest number or letter for OCR correction
def correct_plate_format(ocr_text):
    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ""  # discard if not 7 chars
 
    corrected = []
    for i, char in enumerate(ocr_text):
        if i in (2, 3):  # digit positions
            if char.isdigit():
                corrected.append(char)
            elif char in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[char])
            else:
                return ""
        else:  # letter positions (0, 1, 4, 5, 6)
            if char.isalpha():
                corrected.append(char)
            elif char in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[char])
            else:
                return ""
    return "".join(corrected)


# preprocessing the license plate region before OCR

def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""
 
    # Grayscale + upscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    plate_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
 
    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if ocr_result:
            # Join all segments in case EasyOCR splits the plate into pieces
            candidate = correct_plate_format("".join(ocr_result))
            if candidate and plate_pattern.match(candidate):
                return candidate
    except Exception as e:
        print(f"OCR error: {e}")
    return ""
           


# number plate stablization buffer
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}
last_seen = {}

def get_stable_plate(track_id, new_text, frame_idx):
    last_seen[track_id] = frame_idx
    if new_text:
        plate_history[track_id].append(new_text)
        hist = plate_history[track_id]
        plate_final[track_id] = max(set(hist), key=hist.count)  # majority vote
    return plate_final.get(track_id, "")
 
 
def prune_old_tracks(frame_idx, max_age=150):
    stale = [tid for tid, seen in last_seen.items() if frame_idx - seen > max_age]
    for tid in stale:
        last_seen.pop(tid, None)
        plate_history.pop(tid, None)
        plate_final.pop(tid, None)


# video for inferencing
input_video = r"D:\UTKARSH\CV\License Plate Detection Test_1080p.mp4"
output_video = "output_with_licensev3.mp4"
 
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {input_video}")
 
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
 
CONF_THRESH = 0.3
overlay_h, overlay_w = 150, 400
frame_idx = 0
 
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
 
        clean_frame = frame.copy()  # crops come from here, never from drawn-on frame
 
        # Tracking gives persistent IDs across frames
        results = model.track(frame,imgsz=1280 ,persist=True, conf=CONF_THRESH, verbose=False)
 
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
 
                # Clamp to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
 
                plate_crop = clean_frame[y1:y2, x1:x2]
                text = recognize_plate(plate_crop)
 
                if box.id is not None:
                    track_id = int(box.id[0])
                    stable_text = get_stable_plate(track_id, text, frame_idx)
                else:
                    stable_text = text  # no tracker ID, use raw reading
 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
 
                # Zoomed plate overlay
                plate_zoom = cv2.resize(plate_crop, (overlay_w, overlay_h))
 
                ox1 = min(x1, W - overlay_w)
                ox1 = max(0, ox1)
                oy1 = y1 - overlay_h - 70  # leave room for text above
                text_below = False
                if oy1 < 0:
                    # Not enough room above: place overlay below the box
                    oy1 = min(y2 + 10, H - overlay_h)
                    text_below = True
                oy1 = max(0, oy1)
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w
 
                if oy2 <= H and ox2 <= W:
                    frame[oy1:oy2, ox1:ox2] = plate_zoom
 
                    if stable_text:
                        ty = min(oy2 + 55, H - 10) if text_below else oy1 - 20
                        cv2.putText(frame, stable_text, (ox1, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)  # black outline
                        cv2.putText(frame, stable_text, (ox1, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)  # white fill
 
        if frame_idx % 100 == 0:
            prune_old_tracks(frame_idx)
 
        out.write(frame)
        cv2.imshow("Annotated Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()