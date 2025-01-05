import math
import torch
from ultralytics import YOLO
import cv2
import cvzone

# cap = cv2.VideoCapture(0)
#
# cap.set(3, 1280)
# cap.set(4, 720)

print(torch.cuda.is_available())

cap = cv2.VideoCapture("videos/road.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = model.names

while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            # border
            cvzone.cornerRect(img, bbox)
            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            # class name
            cls = box.cls[0]
            className = classNames[int(cls)]
            cvzone.putTextRect(img, f'{className} {conf}',(x1,y1-20))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
