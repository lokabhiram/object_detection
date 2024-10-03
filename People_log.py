import cv2
import time
import pandas as pd
import os  # Import the os module
from yolov5 import YOLOv5

# Load the YOLOv5 model (suppress FutureWarning)
model = YOLOv5("yolov5s.pt")

# Usage (replace with your camera details)
camera_ip = "192.168.128.10"
port = 554
channel = 1
stream_type = "0-mainstream"

# Initialize camera capture with increased timeout
camera = cv2.VideoCapture(
    f"rtsp://admin:admin123@{camera_ip}:{port}/avstream/channel={channel}/stream={stream_type}.sdp"
)
camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # Set timeout to 60 seconds

# Set desired frame width and height
frame_width = 1200
frame_height = 800

# Set camera frame width and height (might not be effective for all cameras)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# CSV file path
csv_file = "entry_log.csv"

while True:
    # Capture frame from camera
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture frame from camera.")
        break  # Exit the loop if frame capture fails

    # Resize the frame (if needed, even if camera settings don't take effect)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Perform object detection using the predict method
    results = model.predict(frame)

    # Check for object detection (with visualization and lowered confidence threshold)
    for *xyxy, conf, cls in results.pandas().xyxy[0].itertuples(index=False):
        if conf > 0.4:  # Lowered confidence to 0.4
            # Log entry to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get the object label
            if isinstance(cls, (int, float)):
                try:
                    object_label = model.model.names[int(cls)]
                except IndexError:
                    print(f"Error: Class index {int(cls)} out of range for model.model.names")
                    object_label = f"Class {int(cls)}"
            else:
                object_label = cls 

            data = {"Timestamp": [timestamp], "Object Detected": [True], "Object Label": [object_label]}
            df = pd.DataFrame(data)
            df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)
            print(f"{object_label} detected at {timestamp}")

            # Draw bounding box and label
            label = f"{object_label} {conf:.2f}"
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame (optional)
    cv2.imshow("Camera Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()