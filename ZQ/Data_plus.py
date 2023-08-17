from osgeo import gdal
import numpy as np
import tifffile as tiff
import os

# pic_path1 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\G_tif\0%/'  # 需要修改的图片路径
pic_path1 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\RGB\0%/'  # 需要修改的图片路径
save_path1 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\TEMP\O\0%/'

# pic_path2 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\G_tif\10~20%/'  # 需要修改的图片路径
pic_path2 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\RGB\10~20%/'  # 需要修改的图片路径
save_path2 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\TEMP\O\10~20%/'

# pic_path3 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\G_tif\50~60%/'  # 需要修改的图片路径
pic_path3 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\RGB\50~60%/'  # 需要修改的图片路径
save_path3 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\TEMP\O\50~60%/'

# pic_path4 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\G_tif\80%/'  # 需要修改的图片路径
pic_path4 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\RGB\80%/'  # 需要修改的图片路径
save_path4 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\TEMP\O\80%/'

pic_path5 = r'E:\STUDYCONTENT\Pycharm\classification-pytorch-main\datasets\train\0%/'  # 需要修改的图片路径
save_path5 = r'E:\STUDYCONTENT\Pycharm\classification-pytorch-main\datasets\train\0%/'


def Roat180(path, save_path):
    Img_path = os.listdir(path)  # 获取原路径下的图片
    for i in range(len(Img_path)):
        name = Img_path.pop()
        image = gdal.Open(path + name)
        image = image.ReadAsArray()
        image = np.array(image).astype(np.float32)
        image = image[::-1]
        name = 'Roat180' + name
        tiff.imwrite(save_path + name, image)


def Roat90(path, save_path):
    Img_path = os.listdir(path)  # 获取原路径下的图片
    len_path = len(Img_path)
    # print(Img_path)
    for i in range(len_path):
        name = Img_path.pop()
        image = gdal.Open(path + name)
        image = image.ReadAsArray()
        image = np.array(image).astype(np.float32)
        image = np.rot90(image)
        name = 'Roat90' + name
        tiff.imwrite(save_path + name, image)


def Roat90_180(path, save_path):
    Img_path = os.listdir(path)  # 获取原路径下的图片
    # print(path)
    for i in range(len(Img_path)):
        name = Img_path.pop()
        image = gdal.Open(path + name)
        image = image.ReadAsArray()
        image = np.array(image).astype(np.float32)
        image = np.rot90(image)
        image = image[::-1]
        name = 'Roat90_180' + name
        # print(name)
        tiff.imwrite(save_path + name, image)


def Roat180_90(path, save_path):
    Img_path = os.listdir(path)  # 获取原路径下的图片
    for i in range(len(Img_path)):
        name = Img_path.pop()
        image = gdal.Open(path + name)
        image = image.ReadAsArray()
        image = np.array(image).astype(np.float32)
        image = image[::-1]
        image = np.rot90(image)
        name = 'Roat180_90' + name
        tiff.imwrite(save_path + name, image)


Roat90(pic_path1, save_path1)
Roat90(pic_path2, save_path2)
Roat90(pic_path3, save_path3)
Roat90(pic_path4, save_path4)
print('done')
Roat180(pic_path1, save_path1)
Roat180(pic_path2, save_path2)
Roat180(pic_path3, save_path3)
Roat180(pic_path4, save_path4)
print('done')
Roat90_180(pic_path1, save_path1)
Roat90_180(pic_path2, save_path2)
Roat90_180(pic_path3, save_path3)
Roat90_180(pic_path4, save_path4)
print('done')
Roat180_90(pic_path1, save_path1)
Roat180_90(pic_path2, save_path2)
Roat180_90(pic_path3, save_path3)
Roat180_90(pic_path4, save_path4)
# Roat90_180(pic_path5, save_path5)
print('done')
