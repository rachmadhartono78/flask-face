from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import io

app = Flask(__name__)

# Load known faces
path = 'lib/attendance'
images = [cv2.imread(f'{path}/{cls}') for cls in os.listdir(path) if cls]
classNames = [os.path.splitext(cls)[0] for cls in os.listdir(path)]
encodeListKnown = [
    face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0]
    for img in images if img is not None
]

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition</title>
    </head>
    <body>
        <h1>Face Recognition</h1>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <script>
            // Access the user's camera
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then((stream) => {
                    video.srcObject = stream;
                })
                .catch((error) => {
                    console.error("Error accessing camera:", error);
                });

            // Send frames to the server every 100ms
            setInterval(() => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Convert frame to JPEG and send to server
                canvas.toBlob((blob) => {
                    fetch('/upload', {
                        method: 'POST',
                        body: blob,
                        headers: {
                            'Content-Type': 'image/jpeg'
                        }
                    }).catch(err => console.error("Error sending frame:", err));
                }, 'image/jpeg');
            }, 100); // Interval in milliseconds
        </script>
    </body>
    </html>
    """

@app.route('/upload', methods=['POST'])
def upload():
    """Receive frames from the client and process them."""
    try:
        # Get the image from the request
        img_bytes = io.BytesIO(request.data)
        img = Image.open(img_bytes)

        # Convert image to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Resize and process the frame for face recognition
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            matchIndex = matches.index(True) if True in matches else -1
            if matchIndex != -1:
                name = classNames[matchIndex].upper()
                print(f"Detected: {name}")
            else:
                print("Unknown face detected")

        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
