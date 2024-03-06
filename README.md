# 以图搜图

<p align="center">
• 🤖 <a href="https://modelscope.cn/models/iic/cv_resnest101_general_recognition/summary" target="_blank">ModelScope</a> • 📃 <a href="https://milvus.io/" target="_blank">Milvus</a>  
</p>

## 介绍

本文档旨在介绍如何使用 ModelScope 中的通用领域模型（iic/cv_resnest101_general_recognition）以及搭配 Milvus 向量数据库实现以图搜图的功能。这项技术能够有效地从大量图片中迅速识别出与目标图片相似的图片，为图像检索任务提供了一种高效的解决方案

### Milvus

- **数据库类型：** Milvus 向量数据库
- **数据库特点：** Milvus 是一个开源的向量相似性检索引擎，专注于高性能的向量相似性搜索。它支持多种相似性搜索算法，并提供了可扩展的架构，适用于大规模的向量数据存储和检索。
- **数据库描述：** Milvus 向量数据库能够帮助用户轻松应对海量非结构化数据（图片/视频/语音/文本）检索。单节点 Milvus 可以在秒内完成十亿级的向量搜索，分布式架构亦能满足用户的水平扩展需求。

### cv_resnest101_general_recognition

- **模型名称：** iic/cv_resnest101_general_recognition
- **模型类型：** 视觉分类
- **模型描述：** 该模型基于 ResNeSt101 架构，经过大规模数据集的预训练和精调，具有较强的图像识别能力。它能够识别通用领域中的各种物体、场景和图案

## 代码使用流程

1. **下载模型：** 

   ```
   python model_structure/modescope_init.py
   ```

   记录好模型下载后的路径，一般情况下模型路径为：

   ```
   Linux_ModelFile="/root/.cache/modelscope/hub/damo/cv_resnest101_general_recognition/pytorch_model.pt"
   Windows_ModelFile='C:\\Users\\Administrator\\.cache\\modelscope\\hub\\damo\\cv_resnest101_general_recognition\\pytorch_model.pt'
   ```

2. **建立向量库：** 

   ```
   python milvus_manage/mlivus_create.py --host 192.168.10.60 
   ```

   host 为 milvus 数据库所在服务器地址

3. **图像转向量：** 

   ```
   pyhon Image_vectorization.py  --host 192.168.10.60  --data data
   ```

   data目录中直接存放图片

4. **搜索功能使用：** 

   ```
   python webui.py --host 192.168.10.60 --server_port 9090 --limit 4
   ```

   limit限制搜索图片数量

## 结果展示

![1709712838229](example_image\1709712838229.png)

## 模型部署环境

待更新

## Milvus部署及使用

待更新

### Milvus可视化工具Attu

待更新

## Data数据示例

待更新