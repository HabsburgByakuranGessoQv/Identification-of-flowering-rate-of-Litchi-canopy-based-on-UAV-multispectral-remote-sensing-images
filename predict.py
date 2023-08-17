from osgeo import gdal
from classification import Classification
import os
import numpy as np
import cv2

def Pression(path):
    pic_path = path  # 需要修改的图片路径
    pic_name = os.listdir(pic_path)#获取原路径下的图片\
    img_num =  len(pic_name)
    pre_false = 0
    for i in range(len(pic_name)):
        name = pic_name.pop()
        try:
            image = cv2.imread(pic_path + name)
            # image = image.ReadAsArray()
            # image = image.transpose(1, 2, 0)
            image= np.array(image).astype(np.float32)
            image = (np.nan_to_num(image))/255
        except:
            print('Open Error! Try again!')
            continue
        else:

            class_name = classfication.detect_image(image)
            class_ture = pic_path.split('/')[2]
            if class_name != class_ture:
                pre_false= pre_false+1
    ture_rate = 1-pre_false/img_num
    false_rate = pre_false/img_num

    return pre_false,img_num,ture_rate,false_rate

path1='datasets/test/0%/'
path2='datasets/test/10%~20%/'
path3='datasets/test/50%~60%/'
path4='datasets/test/over_80%/'

pic_path = r'E:\STUDYCONTENT\Pycharm\classification-pytorch-main-rgb\logs/'
pic_name = os.listdir(pic_path)
tem1 = 0
tem2 = 0
best_top1score = 0
best_top1model = ''
best_avescore = 0
best_avemodel = ''
for i in range(len(pic_name)):
    classfication = Classification(i)
    pre_false1,img_num1,ture_rate1,false_rate1 = Pression(path1)
    pre_false2,img_num2,ture_rate2,false_rate2 = Pression(path2)
    pre_false3,img_num3,ture_rate3,false_rate3 = Pression(path3)
    pre_false4,img_num4,ture_rate4,false_rate4 = Pression(path4)
    if 1-(pre_false1+pre_false2+pre_false3+pre_false4)/(img_num1+img_num2+img_num3+img_num4)>tem1:
        tem1 = 1-(pre_false1+pre_false2+pre_false3+pre_false4)/(img_num1+img_num2+img_num3+img_num4)
        best_top1score = tem1
        best_top1model = pic_name[i]
    if (ture_rate1+ture_rate2+ture_rate3+ture_rate4)/4>tem2:
        tem2 = (ture_rate1+ture_rate2+ture_rate3+ture_rate4)/4
        best_avescore = tem2
        best_avemodel = pic_name[i]
    # print("-" * 10+"0%预测情况"+"-" * 10)
    # print('预测错误数量：',pre_false1)
    # print('图片总数：',img_num1)
    # print('预测正确率：',str(ture_rate1*100)+'%')
    # print('预测错误率：',str(false_rate1*100)+'%')
    #
    # print("-" * 10+"10%~20%预测情况"+"-" * 10)
    # print('预测错误数量：',pre_false2)
    # print('图片总数：',img_num2)
    # print('预测正确率：',str(ture_rate2*100)+'%')
    # print('预测错误率：',str(false_rate2*100)+'%')
    #
    # print("-" * 10+"50%~60%预测情况"+"-" * 10)
    # print('预测错误数量：',pre_false3)
    # print('图片总数：',img_num3)
    # print('预测正确率：',str(ture_rate3*100)+'%')
    # print('预测错误率：',str(false_rate3*100)+'%')
    #
    # print("-" * 10+"over_80%预测情况"+"-" * 10)
    # print('预测错误数量：',pre_false4)
    # print('图片总数：',img_num4)
    # print('预测正确率：',str(ture_rate4*100)+'%')
    # print('预测错误率：',str(false_rate4*100)+'%')
    #
    print("-" * 10+"总预测情况"+"-" * 10)
    print('预测准确率：',str((1-(pre_false1+pre_false2+pre_false3+pre_false4)/(img_num1+img_num2+img_num3+img_num4))*100)+'%')
    print('四类平均准确率：',str(((ture_rate1+ture_rate2+ture_rate3+ture_rate4)/4)*100)+'%')

print('\033[1;31;42m')
print("-"*10+"最好的模型"+"-" * 10)
print('top1最高分：', str(best_top1score*100) + '%')
print('top1最好的模型：',best_top1model)
print('ave最高分：',str(best_avescore*100) + '%')
print('ave最好的模型：',best_avemodel)
print('\033[0m')


