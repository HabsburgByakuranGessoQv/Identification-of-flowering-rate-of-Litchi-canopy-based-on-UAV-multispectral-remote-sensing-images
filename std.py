import numpy as np
import os
from osgeo import gdal


num1 = 422261760
num2 = 421113065
num3 = 419596115
num4 = 410767590
num5 = 410767590
num6 = 410767590
mean1 = 0.29842865019998754
mean2 = 0.29291234947708983
mean3 = 0.1590252907804978
mean4 = 0.34087137754443153
mean5 = 0.7678303623294395
mean6 = 0.20349380596780828

dsqs1_all = 0
dsqs2_all = 0
dsqs3_all = 0
dsqs4_all = 0
dsqs5_all = 0
dsqs6_all = 0
dsqs7_all = 0
dsqs8_all =0

channels = 6  # 通道数
pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\Final\OSAVI_NDVI_NDRE\\'  # 需要计算的图片路径
classes = ["0%\\", "10%~20%\\", "50%~60%\\", "over_80%\\"]
for ln in classes:
    act_path = os.path.join(pic_path, ln)
    pic_name = os.listdir(act_path)  # 获取原路径下的图片
    for i in range(len(pic_name)):
        name = pic_name.pop()
        # print(name)
        image = gdal.Open(act_path + name)
        image = image.ReadAsArray()
        image = image.transpose(1, 2, 0)
        image = np.array(image).astype(np.float32)
        image = np.nan_to_num(image)

        for i in range(channels):
            channl = np.array(image[:, :, i]).astype(np.float32)

            if i == 0:
                t = np.where(channl == 0)
                channl[t] = mean1
                channl = (channl - mean1) ** 2
                dsqs1_all = dsqs1_all + np.sum(channl)

            if i == 1:
                t = np.where(channl == 0)
                channl[t] = mean2
                channl = (channl - mean2) ** 2
                dsqs2_all = dsqs2_all + np.sum(channl)

            if i == 2:
                t = np.where(channl == 0)
                channl[t] = mean3
                channl = (channl - mean3) ** 2
                dsqs3_all = dsqs3_all + np.sum(channl)

            if i == 3:
                t = np.where(channl == 0)
                channl[t] = mean4
                channl = (channl - mean4) ** 2
                dsqs4_all = dsqs4_all + np.sum(channl)

            if i == 4:
                t = np.where(channl == 0)
                channl[t] = mean5
                channl = (channl - mean5) ** 2
                dsqs5_all = dsqs5_all + np.sum(channl)

            if i == 5:
                t = np.where(channl == 0)
                channl[t] = mean6
                channl = (channl - mean6) ** 2
                dsqs6_all = dsqs6_all +np.sum(channl)

            if i == 6:
                t = np.where(channl == 0)
                channl[t] = mean7
                channl = (channl - mean7) ** 2
                dsqs7_all = dsqs7_all + np.sum(channl)

            if i == 7:
                t = np.where(channl == 0)
                channl[t] = mean8
                channl = (channl - mean8) ** 2
                dsqs8_all = dsqs8_all + np.sum(channl)

std1 = np.sqrt(dsqs1_all / num1)
std2 = np.sqrt(dsqs2_all / num2)
std3 = np.sqrt(dsqs3_all / num3)
std4 = np.sqrt(dsqs4_all / num4)
std5 = np.sqrt(dsqs5_all / num5)
std6 = np.sqrt(dsqs6_all / num6)
# std7 = np.sqrt(dsqs7_all / num7)
# std8 = np.sqrt(dsqs8_all/ num8)

print("num1 =", num1)
print("num2 =", num2)
print("num3 =", num3)
print("num4 =", num4)
print("num5 =", num5)
print("num6 =", num6)
# print("n7 =", num7)
# print("num8 =", num8)

print("mean1 =", mean1)
print("mean2 =", mean2)
print("mean3 =", mean3)
print("mean4 =", mean4)
print("mean5 =", mean5)
print("mean6 =", mean6)
# print("mean7 =", mean7)
# print("mean8 =", mean8)


print("std1 =", std1)
print("std2 =", std2)
print("std3 =", std3)
print("std4 =", std4)
print("std5 =", std5)
print("std6 =", std6)
# print("std7 =", std7)
# print("std8 =",std8)
