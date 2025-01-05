from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
import mysql.connector
from datetime import datetime, timedelta

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

@app.route('/start', methods=['GET'])
def start_recognition():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    # Timer setup
    start_time = datetime.now()

    while True:
        success, img = cap.read()
        
        # Check if no frame captured for 30 seconds
        if not success:
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() > 30:
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({"status": "failed", "message": "Camera not responding for 30 seconds."})
            continue
        else:
            # Reset timer if frame is successfully read
            start_time = datetime.now()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        current_time = datetime.now()

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                if name in attendance_data:
                    last_seen = attendance_data[name]['last_seen_time']
                    if (current_time - last_seen).total_seconds() / 60.0 > 30:
                        attendance_data[name]['discipline_status'] = 0
                        cursor.execute("UPDATE attendance2 SET discipline_status = 0 WHERE name = %s", (name,))
                    else:
                        attendance_data[name]['discipline_status'] = 1
                        working_hours = (current_time - attendance_data[name]['entry_time']).total_seconds() / 3600.0
                        attendance_data[name]['working_hours'] = working_hours
                        cursor.execute("UPDATE attendance2 SET working_hours = %s WHERE name = %s", (working_hours, name))
                    attendance_data[name]['last_seen_time'] = current_time
                else:
                    attendance_data[name] = {
                        'entry_time': current_time,
                        'last_seen_time': current_time,
                        'discipline_status': 1,
                        'working_hours': 0
                    }
                    cursor.execute(
                        "INSERT INTO attendance2 (name, entry_time, last_seen_time, discipline_status, working_hours) VALUES (%s, %s, %s, %s, %s)",
                        (name, current_time, current_time, 1, 0)
                    )
                conn.commit()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "success", "message": "Recognition ended."})

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
