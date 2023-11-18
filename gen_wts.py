import torch
import struct
import sys 



# Load model
model_path = '/mnt/DD3/Models_yolov8/CUHK-SYSU-tracking480n_2/train3/weights/best.torchscript'
model = torch.jit.load(model_path).float()  # load to FP32
print(model)
model.eval()

f = open(model_path.replace('.torchscript','.wts'), 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
