from ultralytics import YOLO
from ultralytics import settings



if __name__=='__main__':

    model = YOLO('yolov8n.pt',task='detect')

    result = model.train(data='/mnt/DD5/FaceDetection/WIDER_FACES/data.yaml',epochs=500,imgsz=480,batch=24,workers=16,
                         project='/mnt/DD3/Models_yolov8/WIDER_FACES-480n',plots=True)
    