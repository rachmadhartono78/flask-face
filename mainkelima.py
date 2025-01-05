from flask import Flask, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Flask setup
app = Flask(__name__)

# Load known faces
path = 'lib/attendance'
images = []
classNames = []

# Load images and names
for cls in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cls}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

# Encode faces
encodeListKnown = [
    face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0]
    for img in images
]

# Video capture setup
cap = cv2.VideoCapture(0)

def generate_frames():
    frame_count = 0  # Counter for skipping frames

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Resize and convert frame for face recognition
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Face detection and encoding
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                top, right, bottom, left = [v * 4 for v in faceLoc]  # Scale back
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                top, right, bottom, left = [v * 4 for v in faceLoc]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Encode frame to JPEG format with reduced quality
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()

        # Send frame as part of the MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream', methods=['GET'])
def stream():
    """Endpoint for streaming video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance', methods=['GET'])
def get_attendance():
    """Dummy attendance endpoint."""
    return jsonify({"status": "success", "message": "Attendance tracking is active."})

@app.route('/reset', methods=['POST'])
def reset_attendance():
    """Reset attendance data."""
    return jsonify({"status": "success", "message": "Attendance data reset."})

if __name__ == '__main__':
    app.run(debug=True)