import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

shop_seg = pipeline(
        Tasks.shop_segmentation, 
        model='damo/cv_vitb16_segmentation_shop-seg'
        )
def seg(image):
    if isinstance(image, str):
        img =  Image.open(os.path.join(image))
        img = ImageOps.exif_transpose(img)
        image = img.convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Invalid input type. Must be a path or an image object.")


    segmentation_result = shop_seg(image)


    original_image =  np.array(image)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    masks = segmentation_result['masks']
    masks_3ch = cv2.merge([masks, masks, masks])
    masked_image = cv2.bitwise_and(original_image, masks_3ch)

    gray_background = np.full_like(original_image, fill_value=128)
    final_image = np.where(masks_3ch == 255, masked_image, gray_background)


    x, y, w, h = cv2.boundingRect(masks)
    cropped_image = final_image[y:y+h, x:x+w]

    return Image.fromarray(cropped_image)

if __name__ == "__main__":
    import os
    from PIL import Image, ImageOps

    img_path = "example_image/异形灯具.jpg"
    img =  Image.open(os.path.join(img_path))
    img_opj = seg(img)
    img_opj.save("model/segmentation_model/异形灯具_分割.jpg")