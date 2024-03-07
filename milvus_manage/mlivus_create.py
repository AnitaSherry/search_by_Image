import argparse
from pymilvus import connections, db
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, utility

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="192.168.10.60", help="数据库IP地址")
parser.add_argument("--database_name", default="image_vector_db_windows", help="milvus数据库名称")
parser.add_argument("--collection_name", default="antique_vector_windows", help="milvus集合名称")
parser.add_argument("--index_name", default="embeding", help="milvus集合名称")
parser.add_argument("--embeding_dim", default=2048, help="向量模型输出维度,这里为resnet101")
parser.add_argument("--index_type", choices=["FLAT", "IVF_FLAT"],default="IVF_FLAT", help="索引类型")
parser.add_argument("--metric_type", choices=["L2","IP","JACCARD"],default="L2", help="度量类型")
parser.add_argument("--nlist", default=128, help="图像分类类别")
args = parser.parse_args()

# 创建连接
conn = connections.connect(host=args.host, port=19530)
# 创建数据库
database = db.create_database(args.database_name)
# 使用数据库
db.using_database(args.database_name)

m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True,)
embeding = FieldSchema(name="embeding", dtype=DataType.FLOAT_VECTOR, dim=args.embeding_dim,)
path = FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256,)
schema = CollectionSchema(
  fields=[m_id, embeding, path],
  description="image to image embeding search",
  enable_dynamic_field=True
)
# 创建合集
collection = Collection(name=args.collection_name, schema=schema, using='default', shards_num=4)


index_params = {
  "metric_type": args.metric_type,
  "index_type": args.index_type,
  "params": {"nlist": args.nlist}
}

# 创建标签
collection.create_index(field_name=args.index_name,index_params=index_params)

utility.index_building_progress(args.collection_name)
print("创建完毕")