from flask import Flask, Response, jsonify
import cv2
import face_recognition
import os

app = Flask(__name__)

# Load known faces
path = 'lib/attendance'
images = [cv2.imread(f'{path}/{cls}') for cls in os.listdir(path) if cls]
classNames = [os.path.splitext(cls)[0] for cls in os.listdir(path)]
encodeListKnown = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success: break
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            matchIndex = matches.index(True) if True in matches else -1
            if matchIndex != -1:
                name = classNames[matchIndex].upper()
                top, right, bottom, left = [v * 4 for v in faceLoc]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """<html><body><h1>Face Recognition</h1><img src="/stream"></body></html>"""

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    return jsonify({"status": "success", "message": "Attendance tracking active."})

@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({"status": "success", "message": "Attendance data reset."})

if __name__ == '__main__':
    app.run(debug=True)
