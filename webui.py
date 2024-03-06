import gradio as gr
import argparse
from net_helper import NetHelper
from PIL import Image
from resnet101_embding.embding import resnet_embeding
from milvus_manage.milvus_operator import MilvusOperator
import time

blank_ = Image.new("RGB", (215, 300))

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
    results = antique_image.search_data(imput_embeding,args.limit)
    # print("milvus向量库搜索时间：", time.time()-start_time, "秒")
    pil_images = [Image.open(result['path']) for result in results]
    # print("查询结果数量：",len(pil_images))
    # print("查询总时间：", time.time()-start_time_0, "秒")
    return pil_images

    
def calculate_rows_and_cols(num_images):
    sqrt_num_images = int(num_images ** 0.5)
    num_cols = sqrt_num_images
    num_rows = num_images // num_cols
    if num_images % num_cols != 0:
        num_rows += 1
    return num_rows, num_cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="192.168.10.60", help="本机IP地址") # 192.168.10.60
    parser.add_argument("--database_name", default="image_vector_db_windows", help="milvus数据库名称")
    parser.add_argument("--collection_name", default="antique_vector_windows", help="milvus集合名称")
    parser.add_argument("--metric_type", choices=["L2","IP","JACCARD"],default="L2", help="度量类型")
    parser.add_argument("--limit", default=4, help="搜索topk")

    args = parser.parse_args()

    antique_image = MilvusOperator(args.host, args.database_name, args.collection_name, args.metric_type)

    net_helper = NetHelper()

    num_rows, num_cols = calculate_rows_and_cols(args.limit)

    app = gr.Interface(
        fn=image_search,
        inputs=gr.inputs.Image(type="pil", source='upload'),
        outputs=gr.Gallery(label="搜索结果").style(height='auto',columns=2),
        title="AnitaSherry/search_by_image",
        theme="default",
        description="如果效果不错帮忙点个stat呦！"
    )

    ip_addr = net_helper.get_host_ip()
    app.queue(concurrency_count=8).launch(show_api=False, share=True, server_name=ip_addr, server_port=9090)
