import torch
import numpy as np
import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['OnRamp','Intersect','Express','UrbMixed'], default='OnRamp')
parser.add_argument('--delay', type=int, choices=[25,30,35,40,45], default=30)
parser.add_argument('--method', type=str, choices=['ours', 'base'], default='ours')
parser.add_argument('--source', type=str, default='datasets/OnRamp/')
parser.add_argument('--label', type=str, default='datasets/labels/OnRamp_label/')
parser.add_argument('--result_path', type=str, default=None)

args = parser.parse_args()
data_param = args.data
delay_param = args.delay
method = args.method
result_path = args.result_path
if result_path is None:
    save = False
else:
    save = True
source = args.source
label_path = args.label


feature_quality_level = 0.2
downsample = 5
FPS = 30


def infer_frame_size(source_path, dataset_name):
    default_size = (1280, 720) if dataset_name == 'Express' else (1920, 1080)
    path = Path(source_path)

    if not path.exists():
        return default_size

    image_candidates = []
    if path.is_file():
        image_candidates = [path]
    else:
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_candidates.extend(sorted(path.glob(ext)))
            image_candidates.extend(sorted(path.glob(ext.upper())))

    for image_path in image_candidates:
        frame = cv2.imread(str(image_path))
        if frame is not None:
            height, width = frame.shape[:2]
            return width, height

    return default_size


fw, fh = infer_frame_size(source, data_param)

data_n = 0

data_l = {'OnRamp':0, 'Intersect':1, 'Express':2, 'UrbMixed':3}
weight_path_l = ['weights/Intersect.pth','weights/OnRamp.pth','weights/Express.pth','weights/UrbMixed.pth']
WINDOW = delay_param
weight_path = weight_path_l[data_l[data_param]]


#device
if 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#detector
imgsz = (fh, fw)
detect_model = 'apps/yolov5/weights/yolov5m.pt'

params = {}
with open('data/OnRamp.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        if 'np.array' in value:
            value = value.replace('np.array', '').replace('(', '').replace(')', '')
            value = np.array(eval(value))
        else:
            value = eval(value)
        params[key] = value

