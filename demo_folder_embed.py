import shutil
import torch 
import cv2 
import numpy as np 
import glob
import torchvision
import time
import math
import sys 
import json


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression_embedding(
        prediction,
        embed_size = 32,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4 - embed_size)  # number of classes
    nm = prediction.shape[1] - nc - 4 - embed_size
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm+embed_size), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4 + embed_size), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask, embeddings = x.split((4, nc, nm, embed_size), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i], embeddings[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask, embeddings), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + 1  , x[:, 4]  # boxes (offset by class), scores
        boxes[:,0] = boxes[:,0] - boxes[:,2]/2
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,1] = boxes[:,1] - boxes[:,3]/2
        boxes[:,3] = boxes[:,1] + boxes[:,3]  

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output



def load_image(img, img_size):

    h0, w0 = img.shape[:2]  # orig hw
    r = 1
    if isinstance(img_size,int):
        r = img_size / max(h0, w0)  # resize image to img_size
    else:
        r = min([img_size[1]/w0,img_size[0]/h0])
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None,from_targets=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    if not from_targets:
        clip_coords(coords, img0_shape)
    return coords

lut = {}
# lut[0] = "car"
# lut[1] = "truck"    
# lut[2] = "bus" 
# lut[3] = "vanet"
# lut[4] = 'bike'
# lut[5] = 'person'
lut = {0:"vehicle_light",1:"vehicle_heavy",2:"B",3:"P"}
lut = {0:"plate"}
# lut = {'car':0,"truck":1,"truck-L":2,"truck-E":3,"truck-U":4,'bus':5,'vanet':6,'vanet-L':7,'vanet-E':8,'vanet-U':9}
# lut = dict([(lut[k],k) for k in lut.keys()])
# lut = {0:'D',1:'C'}
model_path = '/mnt/DD3/Models_yolov8/CUHK-SYSU-tracking640s_1/train/weights/best.torchscript'
folder_path = '/mnt/DD5/PedestrainDetection/MOT/MOT17/test/MOT17-01-DPM/img1'
save_path = folder_path+'_prediction5'
import os 
if not os.path.exists(save_path):
    os.makedirs(save_path)
device =  'cpu'
print(device)
model = torch.jit.load(model_path).to(device)
# model = attempt_load(model_path).to(device)  # load FP32 model
model.eval()


imgsz = (640,640) 
# mask = cv2.imread(folder_path+'/mask.jpg')
# mask[mask<=10] = 0
# mask[mask>10] = 1
target_embed = np.array([0.2481, -0.0442, -0.1033, -0.3071, -0.0081, -0.2419, -0.0338, -0.0870,
        -0.1469,  0.4624, -0.2982, -0.0597,  0.1023,  0.2847,  0.2858, -0.1125,
         0.0523,  0.1868,  0.0071,  0.1933, -0.0110,  0.0452,  0.1246, -0.0529,
         0.0172,  0.1516,  0.1034,  0.1974,  0.2718, -0.0026,  0.0288,  0.0860])
for n,img_path in enumerate(glob.glob(folder_path+'/*.jpg')):
    # if n%2:
    #     os.remove(img_path)
    #     continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    # try:
    #     img *= mask
    # except:
    #     pass 
    # cv2.imwrite(img_path,img)
    img2, (h0, w0), (h, w) = load_image(img, imgsz)
    
    img2, ratio, pad = letterbox(img2, imgsz, auto=False, scaleup=False)
    shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # img2 = img2.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #img2 = np.ascontiguousarray(img2)
    inputs = torch.from_numpy(img2)
    inputs = inputs.permute(2,0,1)
    inputs = inputs.to(torch.float).to(device)/255.0
    
    inputs = inputs.unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs)[0]
    outputs = non_max_suppression_embedding(outputs.cpu(),32,0.4)[0]

    false_classes = [] 
    image_name = img_path.split('/')[-1]
    # f = open(save_path + '/'+image_name.replace('.jpg','.json'), 'w')
    annot = {"image":image_name,"annotations":[]}

    gn = torch.tensor(shapes[0])[[1, 0, 1, 0]]  # normalization gain whwh
    H,W,_ = img.shape
    write_image= False 
    sim_threshold = 0.9
    for idx in range(outputs.size(0)):
        xywh, conf, cls, embed = outputs[idx,:4].numpy(),outputs[idx,4].item(),outputs[idx,5].item(),outputs[idx,6:]
        
        embed = torch.nn.functional.normalize(embed,p=2.0,dim=-1)
        # print(embed)
        sim = (embed.numpy() * target_embed).sum()
        color = (0,255,0)
        print("Sim = ", sim)
        
        if sim>sim_threshold:
            color = (0,0,255)
            write_image = True 
        if conf>=0.6:

            ant = {"label": str(cls), "coordinates": {"x": int(xywh[0]), "y": int(xywh[1]), "width": int(xywh[2]), "height": int(xywh[3]) }}
            annot["annotations"].append(ant)
            #f.write("%d %f %f %f %f  \n"%(int(cls),xywh[0], xywh[1], xywh[2], xywh[3]))
        x, y, w, h = int(xywh[0]),int(xywh[1]),int(xywh[2]),int(xywh[3])
        img2 = cv2.rectangle(img2, (x-w//2, y-h//2),(x+w//2, y+h//2),color,2)
        cv2.putText(img2,str(sim*100)[:4],(x-w//2, y-h//2),4,0.5,(255,0,0),1)
    if write_image:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + '/'+image_name,img2)
        # 
    # json.dump([annot],f)
    # f.close()

    #cv2.imwrite(save_path + '/'+path.split('/')[-1],img_back)