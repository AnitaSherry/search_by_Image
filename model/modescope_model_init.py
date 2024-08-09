import os
import cv2
import numpy as np
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

def classification_model(input_location):
    general_classification = pipeline(
        Tasks.general_recognition,
        model='damo/cv_resnest101_general_recognition'
        )
    classification_result = general_classification(input_location)
    print(classification_result)

def segmentation_model(input_location):
    image = Image.open(input_location)
    channel_type = image.mode
    directory, filename = os.path.split(input_location)
    name, ext = os.path.splitext(filename)
    black_and_white_background = os.path.join(directory, name + "_黑白分割" + ext)
    pure_gray_background = os.path.join(directory, name + "_灰色背景" + ext)
    pure_gray_up_background = os.path.join(directory, name + "_灰色背景升级" + ext)
    shop_seg = pipeline(
        Tasks.shop_segmentation, 
        model='damo/cv_vitb16_segmentation_shop-seg'
        )
    # 获取分割后结果
    segmentation_result = shop_seg(input_location)

    # 保存黑白背景图像
    cv2.imwrite(black_and_white_background, segmentation_result[OutputKeys.MASKS])

    # 读取原始图像
    original_image = cv2.imread(input_location)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 获取掩码
    masks = segmentation_result['masks']
    masks_3ch = cv2.merge([masks, masks, masks])

    # 将掩码应用到原始图像上
    masked_image = cv2.bitwise_and(original_image, masks_3ch)

    # 创建一个填充值为128的灰色背景
    gray_background = np.full_like(original_image, fill_value=128)

    # 合成最终图像
    final_image = np.where(masks_3ch == 255, masked_image, gray_background)

    # 计算物体的边界框
    x, y, w, h = cv2.boundingRect(masks)

    # 裁剪出物体的矩形区域
    cropped_image = final_image[y:y+h, x:x+w]

    # 保存裁剪后的图像
    if channel_type == "BGR":
        cv2.imwrite(pure_gray_background, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pure_gray_up_background, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    else:
        final_image_pil = Image.fromarray(final_image)
        final_image_pil.save(pure_gray_background)

        cropped_image_pil = Image.fromarray(cropped_image)
        cropped_image_pil.save(pure_gray_up_background)
    
    classification_model(pure_gray_up_background)
    print(f"原始图片位于{input_location}")
    print(f"请在{pure_gray_up_background}查看处理后的图片")

if __name__ == "__main__":
    input_location = "example_image/异形灯具.jpg"
    classification_model(input_location)
    segmentation_model(input_location)