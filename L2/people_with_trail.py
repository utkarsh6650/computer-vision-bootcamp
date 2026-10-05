import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("walking.mp4")

id_map = {}  # Dictionary to map object IDs to their trail points   
next_id = 1

trail = defaultdict(lambda : deque(maxlen=30))
appear  = defaultdict(int)

while True:
    ret,frame = cap.read()
    res = model.track(frame,classes=[0],persist= True,verbose = False)
    annoted_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()
        for box, oid in zip(boxes,ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trail[oid].append((cx, cy))
            appear[oid] += 1

            if appear[oid] > 5:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))
                cv2.rectangle(annoted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annoted_frame, f"ID: {sid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annoted_frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.show("Annotated Frame with Object Tracking and Trails", annoted_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
