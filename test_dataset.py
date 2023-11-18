from ultralytics.data.dataset import YOLODataset,YoloTrackingDataset
import cv2 
from matplotlib import pyplot as plt 
import torch 
from tqdm import tqdm
import numpy as np 
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, TQDM_BAR_FORMAT, __version__, callbacks, clean_url, colorstr,
                               emojis, yaml_save)
from ultralytics.cfg import get_cfg, get_save_dir



if __name__=='__main__':
    images_path = '/mnt/DD5/PedestrainDetection/CUHK-SYSU/images'
    imgsz = 320
    data_conf = '/mnt/DD5/PedestrainDetection/CUHK-SYSU/data.yaml'
    data = check_det_dataset(data_conf)
    data['embed_size'] = 32
    dataset = YoloTrackingDataset(img_path=images_path,
            imgsz=imgsz,
            batch_size=1,
            augment=True,  # augmentation
            use_segments=False,
            use_keypoints=False,
            data=data
    )
    result = dataset.__getitem__(1) 
    print(result.keys())