import os
import argparse
from PIL import Image, ImageSequence
from resnet101_embding.embding import resnet_embeding
from milvus_manage.milvus_operator import MilvusOperator
import json
from config import *
import re
from pypinyin import pinyin, Style
from tqdm import tqdm
import time
def extract_and_convert(text):
    # 提取中文、英文和数字字符
    extracted_text = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
    extracted_text = '_'.join(extracted_text)
    # 将中文字符转换为拼音
    pinyin_text = pinyin(extracted_text, style=Style.NORMAL)
    pinyin_text = '_'.join([item[0] if isinstance(item, list) else item for item in pinyin_text])
    
    return pinyin_text

def update_image_vector(new_collection_name, data_path, operator: MilvusOperator):
    image_data = []
    file_list = []

    for root, dirs, files in os.walk(data_path):
        abs_root = os.path.abspath(root)
        for file in files:

            full_path = os.path.join(abs_root, file)
            file_list.append((full_path, file))

    for full_path, file in tqdm(file_list, desc=new_collection_name):

        embeding = resnet_embeding(full_path)

        image_data.append({"file_name": file,"path":full_path, "embedding": embeding[0].detach().cpu().numpy().tolist()})

    insert_info = operator.insert_data(new_collection_name,image_data)
    return insert_info
def batch_import(data_path):
    upload_image = MilvusOperator()
        # 确保提供的路径存在
    if not os.path.exists(data_path):
        print(f"错误：路径 '{data_path}' 不存在。")
        return 
    # 获取文件夹中的所有项目
    items = os.listdir(data_path)
    
    # 过滤出子文件夹
    subfolders = [item for item in items if os.path.isdir(os.path.join(data_path, item))]
    if not subfolders:
        if len(items) == 0:
            print(f"错误：'{data_path}' 中无图像文件。")
            return 
        
        info = update_image_vector(collection_name, data_path, upload_image)
        if info:
            print(info)
        else:
            print("上传失败")
    else:
        record_collection_name = dict()
        for subfolder in subfolders:
            subfolder_path = os.path.join(data_path, subfolder)
            converted_subfolder = extract_and_convert(subfolder)
            record_collection_name[converted_subfolder] = subfolder
            info = update_image_vector(converted_subfolder, subfolder_path, upload_image)
            if info:
                print(info)
            else:
                print("上传失败")
        with open('record_collection_name.json', 'w', encoding='utf-8') as json_file:
            json.dump(record_collection_name, json_file, ensure_ascii=False, indent=4)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="图片存储地址")
    args = parser.parse_args()

    batch_import(args.data)
 