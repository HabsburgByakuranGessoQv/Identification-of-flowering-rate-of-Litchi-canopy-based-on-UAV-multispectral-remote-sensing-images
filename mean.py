from osgeo import gdal
import numpy as np
import os

num1 = 0
sum1 = 0

num2 = 0
sum2 = 0

num3 = 0
sum3 = 0

num4 = 0
sum4 = 0

num5 = 0
sum5 = 0

num6 = 0
sum6 = 0

num7 = 0
sum7 = 0

num8 = 0
sum8 = 0

channels = 6  # 通道数
pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\Final\OSAVI_NDVI_NDRE\\'  # 需要计算的图片路径
classes = ["0%\\", "10%~20%\\", "50%~60%\\", "over_80%\\"]
for ln in classes:
    act_path = os.path.join(pic_path, ln)
    pic_name = os.listdir(act_path)  # 获取原路径下的图片
    for a in range(len(pic_name)):
        name = pic_name.pop()
        # print(name)
        image = gdal.Open(act_path + name)
        image = image.ReadAsArray()
        image = image.transpose(1, 2, 0)
        image = np.array(image).astype(np.float32)
        image = np.nan_to_num(image)

        for i in range(channels):
            num_total_i = np.count_nonzero(image[:, :, i])
            sum_total_i = np.sum(image[:, :, i])
            if i == 0:
                num1 = num1 + num_total_i
                sum1 = sum1 + sum_total_i
            elif i == 1:
                num2 = num2 + num_total_i
                sum2 = sum2 + sum_total_i
            elif i == 2:
                num3 = num3 + num_total_i
                sum3 = sum3 + sum_total_i
            elif i == 3:
                num4 = num4 + num_total_i
                sum4 = sum4 + sum_total_i
            elif i == 4:
                num5 = num5 + num_total_i
                sum5 = sum5 + sum_total_i
            elif i == 5:
                num6 = num6 + num_total_i
                sum6 = sum6 + sum_total_i
            elif i == 6:
                num7 = num7+num_total_i
                sum7 = sum7+sum_total_i
            elif i == 7:
                num8 = num8+num_total_i
                sum8 = sum8+sum_total_i
mean1 = sum1 / num1
mean2 = sum2 / num2
mean3 = sum3 / num3
mean4 = sum4 / num4
mean5 = sum5 / num5
mean6 = sum6 / num6
# mean7 = sum7 / num7
# mean8 = sum8 / num8
# print(num1+num2+num3+num4+num5+num6+num7+num8)
# print(sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8)

print("num1 =", num1)
print("num2 =", num2)
print("num3 =", num3)
print("num4 =", num4)
print("num5 =", num5)
print("num6 =", num6)
# print("num7 =", num7)
# print("num8 =", num8)

print("mean1 =", mean1)
print("mean2 =", mean2)
print("mean3 =", mean3)
print("mean4 =", mean4)
print("mean5 =", mean5)
print("mean6 =", mean6)
# print("mean7 =", mean7)
# print("mean8 =", mean8)
