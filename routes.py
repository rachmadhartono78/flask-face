from flask import render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import os

# Load known faces
path = 'lib/attendance'
images = []
classNames = []

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

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    """Receive frame from the client."""
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No frame received"}), 400

    frame = request.files['frame'].read()
    nparr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize frame for processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Face detection and encoding
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            return jsonify({"status": "success", "name": name})

    return jsonify({"status": "success", "name": "Unknown"})
