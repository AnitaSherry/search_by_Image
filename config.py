# 模型文件地址
EMB_Linux_ModelFile="/root/.cache/modelscope/hub/damo/cv_resnest101_general_recognition/pytorch_model.pt"
EMB_Windows_ModelFile='C:\\Users\\Administrator\\.cache\\modelscope\\hub\\damo\\cv_resnest101_general_recognition\\pytorch_model.pt'
embeding_dim = 2048

# milvus数据库
## 数据名称
database_name = "image_vector_db"

## 集合名称
collection_name = "light_vector"

## 索引名称
index_name = "embeding"

## 图像分类类别范围1~65536
nlist = 128

## metric_type 可选"IVF_FLAT"
metric_type = "L2"

## index_type 可选"IP","JACCARD"
index_type = "IVF_FLAT"






