from osgeo import gdal
import numpy as np
import tifffile as tiff
import os
import shutil


def Roat180(path, save_path):
    Img_path = os.listdir(path)  # 获取原路径下的图片
    for i in range(len(Img_path)):
        name = Img_path.pop()
        pic_path = os.path.join(path, name)
        image = gdal.Open(pic_path)
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
        pic_path = os.path.join(path, name)
        image = gdal.Open(pic_path)
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
        pic_path = os.path.join(path, name)
        image = gdal.Open(pic_path)
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
        pic_path = os.path.join(path, name)
        image = gdal.Open(pic_path)
        image = image.ReadAsArray()
        image = np.array(image).astype(np.float32)
        image = image[::-1]
        image = np.rot90(image)
        name = 'Roat180_90' + name
        tiff.imwrite(save_path + name, image)


classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
pic_path = r'E:\STUDYCONTENT\Pycharm\AlexNet\flowers\\'
save_path = r'E:\STUDYCONTENT\Pycharm\AlexNet\flowersPlus\\'

for flower in classes:
    flower_path = os.path.join(pic_path, flower)
    flower_save = os.path.join(save_path, flower)
    flower_list = os.listdir(flower_path)
    if os.path.exists(flower_save) is not True:
        os.mkdir(flower_save)
    # copy
    for elem in flower_list:
        elem_flower = os.path.join(flower_path, elem)
        elem_save = os.path.join(flower_save, elem)
        shutil.copy(elem_flower, elem_save)
    # DataPlus
    Roat90(flower_path, flower_save)
    Roat180(flower_path, flower_save)
    Roat90_180(flower_path, flower_save)
    Roat180_90(flower_path, flower_save)
    print('class{:10} Amplified'.format(flower))
