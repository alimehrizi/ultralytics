from ultralytics import YOLO
from ultralytics import settings



if __name__=='__main__':

    model = YOLO('ultralytics/cfg/models/v8/yolov8-track.yaml',task='detrack')

    result = model.train(data='/mnt/DD5/PedestrainDetection/CUHK-SYSU/data.yaml',epochs=500,imgsz=640,batch=16,workers=16,
                         project='/mnt/DD3/Models_yolov8/CUHK-SYSU-tracking640s_1',plots=True)
    