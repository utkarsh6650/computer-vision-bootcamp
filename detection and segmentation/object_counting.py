import cv2
import numpy as np
from ultralytics import YOLO 

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

unique_ids = set()  # Set to store unique object IDs

while True:
    ret, frame = cap.read()
    results = model.track(frame, classes = [0,39], persist = True,verbose = False)  # Use the 'track' method for tracking
    annotated_frame  = results[0].plot()

    if results[0].boxes.id is not None:  # Check if there are any detected objects
        ids = results[0].boxes.id.cpu().numpy()  # Get the IDs of detected objects
        for oid in ids:
            unique_ids.add(oid)  # Add the object ID to the set
        cv2.putText(annotated_frame, f"Unique Objects: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video Feed with Object Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()