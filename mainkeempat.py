from flask import Flask, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
import mysql.connector
from datetime import datetime

# Flask setup
app = Flask(__name__)

# Database setup
db_config = {'host': 'localhost', 'user': 'root', 'password': '', 'database': 'keyperformance'}
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Load known faces
path = 'lib/attendance'
images = []
classNames = []
for cls in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cls}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])
encodeListKnown = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]

attendance_data = {}
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Resize frame for faster processing
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        # Face detection and recognition
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                top, right, bottom, left = [v * 4 for v in faceLoc]  # Scale back to original size
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                top, right, bottom, left = [v * 4 for v in faceLoc]  # Scale back to original size
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame as part of an MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream', methods=['GET'])
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance', methods=['GET'])
def get_attendance():
    return jsonify(attendance_data)

@app.route('/reset', methods=['POST'])
def reset_attendance():
    global attendance_data
    attendance_data = {}
    return jsonify({"status": "success", "message": "Attendance data reset."})

if __name__ == '__main__':
    app.run(debug=True)

