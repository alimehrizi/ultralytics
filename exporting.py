from ultralytics import YOLO
from ultralytics import settings
import struct


if __name__=='__main__':
    model_path = '/mnt/DD3/Models_yolov8/CrowdHuman-480n/train/weights/best.pt'
    model = YOLO(model_path)
    print(model.model)
    model.export(format='onnx',opset=14,dynamic=True)
    print(model)

    f = open(model_path.replace('.pt','.wts'), 'w')
    f.write('{}\n'.format(len(model.model.state_dict().keys())))
    for k, v in model.model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
