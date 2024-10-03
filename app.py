from flask import Flask, render_template, Response
import cv2
import time
import pandas as pd
import os
from yolov5 import YOLOv5

app = Flask(__name__)

# Load the YOLOv5 model
model = YOLOv5("yolov5s.pt")

# Camera details (replace with yours)
camera_ip = "192.168.128.10"
port = 554
channel = 1
stream_type = "0-mainstream"
camera_url = f"rtsp://admin:admin123@{camera_ip}:{port}/avstream/channel={channel}/stream={stream_type}.sdp"

# Frame size
frame_width = 1200
frame_height = 800

# CSV file path
csv_file = "entry_log.csv"

# Object detection statistics
object_counts = {}

def generate_frames():
    camera = cv2.VideoCapture(camera_url)
    camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        results = model.predict(frame)

        for *xyxy, conf, cls in results.pandas().xyxy[0].itertuples(index=False):
            if conf > 0.4:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Get the object label
                if isinstance(cls, (int, float)):
                    try:
                        object_label = model.model.names[int(cls)]
                    except IndexError:
                        object_label = f"Class {int(cls)}"
                else:
                    object_label = cls

                # Update object counts
                if object_label in object_counts:
                    object_counts[object_label] += 1
                else:
                    object_counts[object_label] = 1

                data = {"Timestamp": [timestamp], "Object Detected": [True], "Object Label": [object_label]}
                df = pd.DataFrame(data)
                df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)
                print(f"{object_label} detected at {timestamp}")

                # Draw bounding box and label
                label = f"{object_label} {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert frame to JPEG data
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html', object_counts=object_counts)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')