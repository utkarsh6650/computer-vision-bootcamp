import cv2

cap = cv2.VideoCapture(0)

frames = []
gap = 5
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    frames.append(gray)

    # keep buffer at exactly gap+1 frames: trim only when it overflows
    if len(frames) > gap + 1:
        frames.pop(0)

    cv2.putText(frame, f"Frame: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # motion detection: only once buffer is full
    if len(frames) == gap + 1:
        diff = cv2.absdiff(frames[0], frames[-1])
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion = False
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            motion = True
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if motion:
            cv2.putText(frame, "Motion Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(f"motion_frame_{count}.jpg", frame)
            print(f"Motion detected! Frame saved as motion_frame_{count}.jpg")

    cv2.imshow("Motion Detection", frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()