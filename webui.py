import gradio as gr
import argparse
from PIL import Image
from model.embding_model.embding import resnet_embeding
from milvus_manage.milvus_operator import MilvusOperator
import time
from ip_socket import host
from config import *
import json
from model.segmentation_model.segmentation import seg
def image_search(coll_name,limit,topk,image):
    coll_name = r_record_collection_name[coll_name]
    start_time_0 = time.time()
    if image is None:
        return None

    # restnet编码
    start_time = time.time()
    if args.use_seg:
        image = seg(image)
    print("分割时间：", time.time()-start_time, "秒")
    start_time = time.time()
    imput_embeding = resnet_embeding(image)
    print("restnet101编码时间：", time.time()-start_time, "秒")
    imput_embeding = imput_embeding[0].detach().cpu().numpy()
    start_time = time.time()
    results = light_image.search_data(imput_embeding,coll_name,int(limit),int(topk))
    # print(results)
    if results:
        print("milvus向量库搜索时间：", time.time()-start_time, "秒")
        pil_images = [Image.open(result['entity']['path']) for result in results]
        # print("查询结果数量：",len(pil_images))
        print("查询总时间：", time.time()-start_time_0, "秒")
        return pil_images,  "\n\n".join([f"top{top}、**相似度**: {result['distance']}\t**名称**: {result['entity']['file_name']}"for top,result in enumerate(results)])
        # return pil_images
    else:
        return ["example_image/未搜索到.jpg"],"没有匹配到相似的图片"
    

if __name__ == "__main__":
    with open('record_collection_name.json', 'r', encoding='utf-8') as json_file:
        record_collection_name = json.load(json_file)
    r_record_collection_name = {value: key for key, value in record_collection_name.items()}
    parser = argparse.ArgumentParser()

    parser.add_argument("--server_port", default=9090, help="port端口号")
    parser.add_argument("--use_seg", action="store_true", help="是否使用图像分割")
    args = parser.parse_args()

    if args.use_seg:
        maximum = 5000
    else:
        maximum = 2000

    light_image = MilvusOperator(host, database_name)
    choices_collections = light_image.show_collections()
    choices_collections = [record_collection_name[i] for i in choices_collections]
    # print(choices_collections)
    app = gr.Interface(
        fn=image_search,
        inputs=[
            gr.Dropdown(
            choices=choices_collections, 
            label="选择一个选项"),
                        gr.Slider(minimum=1, maximum=20, step=1, value=5, label="选择返回的数量"),
            gr.Slider(minimum=100, maximum=maximum, step=100, value=2000, label="L2相似度阈值（越小相似度越高）"),
            gr.Image(type="pil", sources='upload'),

            ],
        outputs=[gr.Gallery(label="搜索结果"),gr.Markdown(label="相似度")],
        title="AnitaSherry/search_by_image",
        theme="default",
        description="如果效果不错帮忙点个stat呦！"
    )

    app.launch(show_api=False, share=True, server_name=host, server_port=int(args.server_port))
