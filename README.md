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
```
# 拉取仓库
$ git clone https://github.com/AnitaSherry/search_by_Image.git

# 进入目录
$ cd search_by_Image
```
### 模型部署环境

```
pip install modelscope
pip install modelscope[cv] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install docker==6.1.3
pip install docker-compose==1.29.2
pip install gradio==3.50.2
pip install pymilvus==2.3.6
```
其他的缺什么装什么,我的环境装包太多不便分享，每个人都有适合自己的包

1. **下载模型：** 

   ```
   python model/modescope_model_init.py
   ```

   记录好模型下载后的路径，一般情况下模型路径为：

   ```
   Linux_ModelFile="/root/.cache/modelscope/hub/damo/cv_resnest101_general_recognition/pytorch_model.pt"
   Windows_ModelFile='C:\\Users\\Administrator\\.cache\\modelscope\\hub\\damo\\cv_resnest101_general_recognition\\pytorch_model.pt'
   ```
   将得到的路径记录下来，替换./resnet101_embding/embding.py代码中第13行的路径

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

![Alt text](example_image/1709712838229.png)

## Milvus部署及使用

```
mkdir Milvus
cd Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.2.13/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker-compose up -d
sudo docker-compose ps
```

通过命令查看显示信息如下

```
      Name                     Command                  State                            Ports
--------------------------------------------------------------------------------------------------------------------
milvus-etcd         etcd -advertise-client-url ...   Up (healthy)   2379/tcp, 2380/tcp
milvus-minio        /usr/bin/docker-entrypoint ...   Up (healthy)   0.0.0.0:9000->9000/tcp, 0.0.0.0:9001->9001/tcp
milvus-standalone   /tini -- milvus run standalone   Up (healthy)   0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
```

验证连接

```
docker port milvus-standalone 19530/tcp
```

停止Milvus

```
sudo docker-compose down
```

停止后删除数据

```
sudo rm -rf  volumes
```

### docker安装

```
sudo yum install docker
sudo systemctl start docker
sudo systemctl enable docker
sudo docker --version
```

输出示例
```
Docker version 18.09.0, build 172f8da
```
### docker-compose安装

```
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
docker-compose -v
```

输出示例

```
docker-compose version 1.29.2, build unknown
```

### Milvus可视化工具Attu

```
docker run -p 8000:3000  -e MILVUS_URL=0.0.0.0:19530 zilliz/attu:dev
```
https://github.com/zilliztech/attu/issues/415
dev版本可以支持arrch昇腾服务器，本人和Attu官方人员沟通后得到版本，x86系统可以使用v2.3.8版本

#### 进入网页端

启动docker后，在浏览器中访问“http://{your machine IP}:8000”，点击“Connect”进入Attu服务

Milvus Address 填写{your machine IP}:19530

Milvus Database (optional) 如果创建过数据库直接填写数据库名称，如果没有填写default，创建一个名为default的数据库

Milvus Username (optional) 和 Milvus Password (optional) 无需填写，因为默认是关闭鉴权的

## Data数据示例

```
链接：https://pan.baidu.com/s/1eEDYq0oCBxmVRrIhophgCQ?pwd=c50e 
提取码：c50e
```

## 鼓励支持
 点个赞再走呗！比心💞️

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=AnitaSherry/search_by_Image&type=Date)](https://star-history.com/#AnitaSherry/search_by_Image&Date)
