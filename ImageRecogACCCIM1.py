from ultralytics import YOLO
import comet_ml
import torch
import cv2
comet_ml.init()
model = YOLO("yolov8n.pt")
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
results = model.predict(["https://ultralytics.com/images/bus.jpg"],show=True)
cv2.waitKey()
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    original = result.orig_img
