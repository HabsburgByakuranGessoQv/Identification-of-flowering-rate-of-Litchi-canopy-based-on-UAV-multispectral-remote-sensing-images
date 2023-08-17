import tifffile as tiff
import numpy as np
import os
import cv2

# pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B/0%/'  # 需要修改的图片路径
# pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B/10~20%/'  # 需要修改的图片路径
# pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B/50~60%/'  # 需要修改的图片路径
pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B/80%/'  # 需要修改的图片路径
save_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B_tif/80%/'
# save_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B_tif/50~60%/'
# save_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B_tif/10~20%/'
# save_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B_tif/0%/'

pic_name = os.listdir(pic_path)

for i in range(len(pic_name)):
    name = pic_name.pop()
    name = name.split('j')[0]
    image = cv2.imread(pic_path + name + 'jpg', -1)
    image = np.array(image).astype(np.float32)
    # print(image.shape)
    image = image / 255.0  # 归一化
    tiff.imwrite(save_path + name + 'tif', image)
