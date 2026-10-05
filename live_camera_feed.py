import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0) # using the default camera (0)
model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    result = model(frame)
    annotated_frame = result[0].plot()
    cv2.imshow("Live Camera Feed with Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()