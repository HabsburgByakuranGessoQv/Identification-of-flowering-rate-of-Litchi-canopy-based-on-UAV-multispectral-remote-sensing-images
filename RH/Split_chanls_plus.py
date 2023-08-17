import cv2
import os

pic_path = r'D:\APPDATAiD\aliyunpan\CanopyFinal\RGB'  # 需要修改的图片路径
save_path11 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B/'
save_path22 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\G/'
save_path33 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\R/'

file_list = os.listdir(pic_path)
# print(file_list)

for i in file_list:
    pic_dpath = os.path.join(pic_path, i)
    save_path1 = os.path.join(save_path11, i)
    save_path2 = os.path.join(save_path22, i)
    save_path3 = os.path.join(save_path33, i)
    # print(pic_dpath)
    pic_name = os.listdir(pic_dpath)  # 获取原路径下的图片
    # print(pic_name)
    is_Exist = os.path.exists(save_path1)
    if not is_Exist:
        os.makedirs(save_path1)
    is_Exist = os.path.exists(save_path2)
    if not is_Exist:
        os.makedirs(save_path2)
    is_Exist = os.path.exists(save_path3)
    if not is_Exist:
        os.makedirs(save_path3)
    for j in pic_name:
        fin_path = os.path.join(pic_dpath, j)
        image = cv2.imread(fin_path)
        (B, G, R) = cv2.split(image)
        cv2.imwrite(save_path1 + '/' + j, B)
        cv2.imwrite(save_path2 + '/' + j, G)
        cv2.imwrite(save_path3 + '/' + j, R)
