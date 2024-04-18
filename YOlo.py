from ultralytics import YOLO
# load pretrained model
model = YOLO("yolov8s.pt")

results = model(source=0,show=True)
