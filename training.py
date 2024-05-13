from ultralytics import YOLO
from ultralytics import settings



if __name__=='__main__':

    model = YOLO('yolov8m.pt',task='detect')
    hyp = dict() 
    hyp["hsv_h"] =  0.015  # (float) image HSV-Hue augmentation (fraction)
    hyp["hsv_s"] =  0.7  # (float) image HSV-Saturation augmentation (fraction)
    hyp["hsv_v"] =  0.4  # (float) image HSV-Value augmentation (fraction)
    hyp["degrees"] =  0.1  # (float) image rotation (+/- deg)
    hyp["translate"] =  0.1  # (float) image translation (+/- fraction)
    hyp["scale"] =  0.2  # (float) image scale (+/- gain)
    hyp["shear"] =  0.00  # (float) image shear (+/- deg)
    hyp["perspective"] =  0.000  # (float) image perspective (+/- fraction), range 0-0.001
    hyp["flipud"] =  0.0  # (float) image flip up-down (probability)
    hyp["fliplr"] =  0.0  # (float) image flip left-right (probability)
    hyp["mosaic"] =  1.0  # (float) image mosaic (probability)
    hyp["mixup"] =  0.0  # (float) image mixup (probability)
    hyp["copy_paste"] =  0.0  # (float) segment copy-paste (probability)

    result = model.train(data='/home/altex/DATASET/TempData/plates.yaml',epochs=100,imgsz=480,batch=32,workers=20,
                         project='/mnt/DD3/Models_yolov8/PlateDetection-480m',plots=True, hyp=hyp,)
    