import gradio as gr
import argparse
from net_helper import NetHelper
from PIL import Image
from resnet101_embding.embding import resnet_embeding
from milvus_manage.milvus_operator import MilvusOperator
import time

def image_search(image):
    # start_time_0 = time.time()
    if image is None:
        return None

    # restnet编码
    # start_time = time.time()
    imput_embeding = resnet_embeding(image)
    # print("restnet101编码时间：", time.time()-start_time, "秒")
    imput_embeding = imput_embeding[0].detach().cpu().numpy()
    # start_time = time.time()
    results = antique_image.search_data(imput_embeding,int(args.limit))
    # print("milvus向量库搜索时间：", time.time()-start_time, "秒")
    pil_images = [Image.open(result['path']) for result in results]
    # print("查询结果数量：",len(pil_images))
    # print("查询总时间：", time.time()-start_time_0, "秒")
    return pil_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="192.168.10.60", help="数据库IP地址") # 192.168.10.60
    parser.add_argument("--server_port", default=9090, help="port端口号")
    parser.add_argument("--database_name", default="image_vector_db_windows", help="milvus数据库名称")
    parser.add_argument("--collection_name", default="antique_vector_windows", help="milvus集合名称")
    parser.add_argument("--metric_type", choices=["L2","IP","JACCARD"],default="L2", help="度量类型")
    parser.add_argument("--limit", default=4, help="搜索topk")
    

    args = parser.parse_args()

    antique_image = MilvusOperator(args.host, args.database_name, args.collection_name, args.metric_type)

    net_helper = NetHelper()

    app = gr.Interface(
        fn=image_search,
        inputs=gr.inputs.Image(type="pil", source='upload'),
        outputs=gr.Gallery(label="搜索结果").style(height='auto',columns=2),
        title="AnitaSherry/search_by_image",
        theme="default",
        description="如果效果不错帮忙点个stat呦！"
    )

    ip_addr = net_helper.get_host_ip()
    app.queue(concurrency_count=8).launch(show_api=False, share=True, server_name=ip_addr, server_port=int(args.server_port))
