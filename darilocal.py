from flask import Flask, jsonify, Response
import threading
import cv2
import numpy as np
import face_recognition
import os
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Database setup
db_config = {
    'host': os.getenv('DB_HOST', '103.220.113.186'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'keyperformance')
}

def connect_db():
    return mysql.connector.connect(**db_config)

# Load known faces
path = 'lib/attendance'
images = []
classNames = []

logging.info("Loading known faces...")
for cls in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cls}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

encodeListKnown = [
    face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images
]
logging.info(f"Loaded {len(classNames)} known faces.")

attendance_data = {}
process_running = False

def recognition_process():
    global process_running, attendance_data

    logging.info("Starting recognition process...")
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cap = cv2.VideoCapture("udp://0.0.0.0:12345")
        if not cap.isOpened():
            logging.error("UDP stream not accessible!")
            process_running = False
            return

        while True:
            success, img = cap.read()
            if not success:
                logging.warning("Failed to read frame from UDP stream.")
                continue

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
    except Exception as e:
        logging.error(f"Error in recognition process: {e}")
    finally:
        if conn.is_connected():
            conn.close()
        process_running = False
        logging.info("Recognition process stopped.")

@app.route('/start', methods=['GET'])
def start_recognition():
    global process_running

    if process_running:
        return jsonify({"status": "failed", "message": "Recognition process already running."})

    process_running = True
    thread = threading.Thread(target=recognition_process)
    thread.start()

    return jsonify({"status": "success", "message": "Recognition process started."})

@app.route('/attendance', methods=['GET'])
def get_attendance():
    return jsonify(attendance_data)

@app.route('/reset', methods=['POST'])
def reset_attendance():
    global attendance_data
    attendance_data = {}
    return jsonify({"status": "success", "message": "Attendance data reset."})

@app.route('/monitoring', methods=['GET'])
def monitoring_stream():
    """
    Endpoint untuk streaming video dengan deteksi wajah.
    """
    def generate_frames():
        cap = cv2.VideoCapture("udp://0.0.0.0:12345")
        if not cap.isOpened():
            logging.error("UDP stream not accessible for monitoring!")
            yield b''

        while True:
            success, frame = cap.read()
            if not success:
                logging.warning("Failed to read frame for monitoring.")
                break

            # Deteksi wajah
            face_locations = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Encode frame ke format JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
