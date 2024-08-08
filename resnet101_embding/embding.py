import torch
import torchvision 
import sys
sys.path.append('/home/search_by_Image/')
import os.path as osp
from resnet101_embding import resnet
from typing import Any, Dict
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from config import EMB_Linux_ModelFile,EMB_Windows_ModelFile

import platform
os_name = platform.system()
if os_name == "Windows":
    ModelFile=EMB_Windows_ModelFile
elif os_name == "Linux":
    ModelFile=EMB_Linux_ModelFile
else:
    print("当前系统不是 Windows 也不是 Linux")

device = "cuda" if torch.cuda.is_available() else "cpu"

def resnest101(**kwargs):
    model = resnet.ResNet(
        resnet.Bottleneck, [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs)
    return model

def filter_param(src_params, own_state):
    copied_keys = []
    for name, param in src_params.items():
        if 'module.' == name[0:7]:
            name = name[7:]
        if '.module.' not in list(own_state.keys())[0]:
            name = name.replace('.module.', '.')
        if (name in own_state) and (own_state[name].shape
                                    == param.shape):
            own_state[name].copy_(param)
            copied_keys.append(name)

def load_pretrained(model, src_params):
    if 'state_dict' in src_params:
        src_params = src_params['state_dict']
    own_state = model.state_dict()
    filter_param(src_params, own_state)
    model.load_state_dict(own_state)
    model.eval()

def read_image(img_path):
    if isinstance(img_path, str):
        img =  Image.open(os.path.join(img_path))
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
    elif isinstance(img_path, Image.Image):
        img = img_path.convert('RGB')
    else:
        raise ValueError("Invalid input type. Must be a path or an image object.")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])
    img = transform(img)
    return torch.unsqueeze(img, 0).to(device)

def resnet_embeding(ImageFile):
    input_img = read_image(ImageFile)
    return model(input_img)

src_params = torch.load(ModelFile)
model = resnest101().to(device)
load_pretrained(model, src_params)



if __name__ == '__main__':
    ImageFile = "data/方形/1.jpg"

    outputs = resnet_embeding(ImageFile)
    
    print(outputs)
    print(outputs.shape) # [1, 2048]
    outputs = outputs[0].cpu().tolist()
    print(len(outputs))
    print("开始导入")
    from milvus_manage.milvus_operator import MilvusOperator
    upload_image = MilvusOperator()
    from pymilvus import MilvusClient
    from ip_socket import host
    database_name = "image_vector_db"
    client = MilvusClient(uri="http://"+host+":19530",db_name=database_name)
    data = [{'id':1,'file_name': '41.jpg', 'path': 'data/圆形/41.jpg', 'embedding': outputs}]
    upload_image.insert_data(collection_name="yuan_xing",data=data)
    # client.insert(collection_name="yuan_xing",data=data)