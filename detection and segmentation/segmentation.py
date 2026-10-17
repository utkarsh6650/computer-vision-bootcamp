import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # YOLOv8 nano
cap = cv2.VideoCapture(r"D:\UTKARSH\CV\walking.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Could not open video file")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    results = model.track(frame, classes=[0], persist=True, verbose=False)

    annotated_frame = frame.copy()

    for r in results:
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            polygons = r.masks.xy
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()

            for poly, box, person_id in zip(polygons, boxes, ids):
                if len(poly) == 0:
                    continue
                cv2.polylines(annotated_frame, [poly.astype(np.int32)],
                        isClosed=True, color=(0, 255, 0), thickness=2)
                x1,y1 = int(box[0]), int(box[1])
                cv2.putText(annotated_frame, f"ID: {person_id}",
                        (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object tracking with Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Processing completed.")