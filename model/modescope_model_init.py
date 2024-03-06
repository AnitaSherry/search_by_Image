from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

general_recognition = pipeline(
            Tasks.general_recognition,
            model='damo/cv_resnest101_general_recognition')
result = general_recognition('https://pailitao-image-recog.oss-cn-zhangjiakou.aliyuncs.com/mufan/img_data/maas_test_data/dog.png')
print(result)