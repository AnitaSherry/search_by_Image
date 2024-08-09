from pymilvus import connections, db, Collection, utility
from config import *
from ip_socket import host
from pymilvus import MilvusClient,  DataType



class MilvusOperator:
    def __init__(self, host=host, database=database_name):
        self.database = database
        connections.connect(host=host, port=19530,alias="default")
        all_databases = db.list_database()
        if self.database not in all_databases:
            self.database = db.create_database(self.database)
        connections.disconnect(alias="default")
        connections.disconnect(alias=database)
        self.client = MilvusClient(uri="http://"+host+":19530",db_name=database_name)
    def insert_data(self, collection_name, data):
        try:
            if collection_name not in self.client.list_collections():
                print(f"创建新集合{collection_name}")
                schema = MilvusClient.create_schema(enable_dynamic_field=True)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
                schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1024)
                schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embeding_dim)
                schema.verify()

                index_params = self.client.prepare_index_params()
                if len(data)>nlist*10:
                    if len(data)>65536*10:
                        nlist = 65536
                    nlist = len(data)/10
                index_params.add_index(field_name="embedding",
                                    index_type=index_type,
                                    metric_type=metric_type,
                                    params={"nlist": nlist}
                                )   
                             
                self.client.create_collection(collection_name=collection_name,
                                        schema=schema,
                                        index_params=index_params,
                                    )
            
            insert_info = self.client.insert(collection_name=collection_name,data=data)
            return insert_info
        except Exception as e:
            print(f"Error: {e}")
            return 
 
    def search_data(self, embeding,coll_name,limit,topk):
        docs = []
        search_res = self.client.search(collection_name=coll_name,
                                            data=[embeding],
                                            limit=limit,  # Return top 3 results
                                            search_params={"metric_type": metric_type, "params": {}},  # Inner product distance
                                            output_fields=["file_name","path"],  # Return the text field
                                        )
        for i in search_res[0]:
            if i['distance'] < topk:
                docs.append(i)
        # return search_res[0]
        return docs

    def drop_database(self,database_name):
        connections.connect(host=host, port=19530)
        if database_name in db.list_database():
            for collection_name in self.client.list_collections():
                self.client.drop_collection(collection_name)
            db.drop_database(database_name)
            print(f"{database_name}删除成功")
        else:
            print(f"{database_name}不存在，无法删除")
            print(f"现有数据库为{db.list_database()}")
        connections.disconnect()

    def drop_collections(self,collection_name):
        if collection_name in self.client.list_collections():
            self.client.drop_collection(collection_name)
            print(f"{collection_name}删除成功")
        else:
            print(f"{collection_name}不存在，无法删除")
            print(f"现有集合为{self.client.list_collections()}")
    
    def show_collections(self,):
        return self.client.list_collections()