import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np


# Load YOLO model
model = YOLO("yolov8n.pt")


# Open video
cap = cv2.VideoCapture(r"D:\UTKARSH\CV\walking.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()


# Get video properties for saving annotated output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# COCO classes
# 0 = person
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

classes_to_track = [0, 2, 3, 5, 7]


# Create video writer for annotated output
output = cv2.VideoWriter(
    "annotated_person_vehicle_trail.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Check if output writer opened successfully
if not output.isOpened():
    print("ERROR: Could not create output video")
    cap.release()
    exit()


# Map YOLO tracking IDs to our own sequential IDs
id_map = {}
next_id = 1


# Store trail points for each object
trail = defaultdict(lambda: deque(maxlen=10))


# Count how many frames each object has appeared
appear = defaultdict(int)


while True:

    # Read frame
    ret, frame = cap.read()
    

    # Stop if video has ended
    if not ret:
        break


    # Track persons and vehicles
    res = model.track(
        frame,
        classes=classes_to_track,
        persist=True,
        verbose=False
    )


    # Make a copy for drawing
    annotated_frame = frame.copy()


    # Check whether tracking IDs exist
    if res[0].boxes.id is not None:

        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()
        class_ids = res[0].boxes.cls.cpu().numpy()


        # Process every detected object
        for box, oid, class_id in zip(boxes, ids, class_ids):

            oid = int(oid)
            class_id = int(class_id)

            x1, y1, x2, y2 = map(int, box)


            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2


            # Add center point to trail
            trail[oid].append((cx, cy))


            # Count appearances
            appear[oid] += 1


            # Assign our own ID only once
            if appear[oid] > 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1


            # Draw only after assigning an ID
            if oid in id_map:

                sid = id_map[oid]


                # Get class name
                class_name = model.names[class_id]


                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


                # Draw ID and class name
                cv2.putText(
                    annotated_frame,
                    f"{class_name} ID: {sid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


                # Draw center point
                cv2.circle(
                    annotated_frame,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )


                # Draw trail
                points = list(trail[oid])

                for i in range(1, len(points)):

                    cv2.line(
                        annotated_frame,
                        points[i - 1],
                        points[i],
                        (0, 255, 255),
                        2
                    )


    # Save annotated frame
    output.write(annotated_frame)


    # Display frame
    cv2.imshow(
        "Person and Vehicle Tracking",
        annotated_frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()

print("Processing completed.")