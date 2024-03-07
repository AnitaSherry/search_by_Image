import os
import argparse
from PIL import Image, ImageSequence
from resnet101_embding.embding import resnet_embeding
from milvus_manage.milvus_operator import MilvusOperator

 
def update_image_vector(data_path, operator: MilvusOperator):
    idxs, embedings, paths = [], [], []
 
    total_count = 0
    for dir_name in os.listdir(data_path):
        file = os.path.join(data_path, dir_name)

        embeding = resnet_embeding(file)

        idxs.append(total_count)
        embedings.append(embeding[0].detach().cpu().numpy().tolist())
        paths.append(file)
        total_count += 1

        if total_count % 50 == 0:
            data = [idxs, embedings, paths]
            operator.insert_data(data)

            print(f'success insert {operator.coll_name} items:{total_count}', end="\r")
            idxs, embedings, paths = [], [], []
 
        if len(idxs):
            data = [idxs, embedings, paths]
            operator.insert_data(data)
            print(f'success insert {operator.coll_name} items:{total_count}', end="\r")
 
    print(f'finish update {operator.coll_name} items: {total_count}')
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="192.168.10.60", help="数据库IP地址")
    parser.add_argument("--database_name", default="image_vector_db_windows", help="milvus数据库名称")
    parser.add_argument("--collection_name", default="antique_vector_windows", help="milvus集合名称")
    parser.add_argument("--metric_type", choices=["L2","IP","JACCARD"],default="L2", help="度量类型")
    parser.add_argument("--data", default="data", help="图片存储地址")

    args = parser.parse_args()

    antique_image = MilvusOperator(args.host, args.database_name, args.collection_name, args.metric_type)
    update_image_vector(args.data,antique_image)
 