from pymilvus import MilvusClient,db,connections,utility
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ip_socket import host
database_name = "image_vector_db"
connect = connections.connect(host=host, port=19530)
all_databases = db.list_database()
print("现有数据库:",all_databases)

client = MilvusClient(uri="http://"+host+":19530",db_name=database_name)
collections = client.list_collections()
print("现有集合:", collections)
for collection_name in collections:
    client.drop_collection(collection_name)
    print(f"已删除集合: {collection_name}")


if database_name in db.list_database():
    db.drop_database(database_name)
    print(f"{database_name}删除成功")
else:
    print(f"{database_name}不存在，无法删除")
    print(f"现有数据库为{db.list_database()}")

all_databases = db.list_database()
print("现有数据库:",all_databases)
# collections = client.list_collections()
# print("现有集合:", collections)