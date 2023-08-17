import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import get_model_from_name

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes, weights_init
from utils.utils_fit import fit_one_epoch

import os
import shutil

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # ----------------------------------------------------#
    Cuda = True
    # ----------------------------------------------------#
    #   训练自己的数据集的时候一定要注意修改classes_path
    #   修改成自己对应的种类的txt
    # ----------------------------------------------------#
    classes_path = 'model_data/cls_classes.txt'
    # ----------------------------------------------------#
    #   输入的图片大小
    # ----------------------------------------------------#
    input_shape = [224, 224]
    # ----------------------------------------------------#
    #   所用模型种类：
    #   mobilenet、resnet50、vgg16、vit
    #
    #   在使用vit时学习率需要设置的小一些，否则不收敛
    #   可以将最下方的两个lr分别设置成1e-4、1e-5
    # ----------------------------------------------------#
    backbone = "vit"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = ""
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = False
    # ------------------------------------------------------#
    #   获得图片路径和标签
    # ------------------------------------------------------#
    annotation_path = "cls_train.txt"
    # ------------------------------------------------------#
    #   进行训练集和验证集的划分，默认使用10%的数据用于验证
    # ------------------------------------------------------#
    val_split = 0.1
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------#
    num_workers = 4

    # ------------------------------------------------------#
    #   获取classes
    # ------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    print(class_names)
    if backbone != "vit":
        model = get_model_from_name[backbone](num_classes=num_classes, pretrained=pretrained)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        # ------------------------------------------------------#
        #   载入预训练权重
        # ------------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory("logs/")
    # ----------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    # ----------------------------------------------------#
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为1。
    #
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch为总训练世代
    #   提示OOM或者显存不足请调小batch_size
    # ------------------------------------------------------#
    if True:
        # ----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        # ----------------------------------------------------#
        lr = 1e-3
        Batch_size = 2
        Init_Epoch = 0
        Freeze_Epoch = 100

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DataGenerator(lines[:num_train], input_shape, True)
        val_dataset = DataGenerator(lines[num_train:], input_shape, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=detection_collate)
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        # ----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        # ----------------------------------------------------#
        lr = 1e-3
        Batch_size = 2
        Freeze_Epoch = 0
        Epoch = 0

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DataGenerator(lines[:num_train], input_shape, True)
        val_dataset = DataGenerator(lines[num_train:], input_shape, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=detection_collate)
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if Freeze_Train:
            model.Unfreeze_backbone()

        for epoch in range(Freeze_Epoch, Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Epoch, Cuda)
            lr_scheduler.step()


    move_path = r'E:\STUDYCONTENT\Pycharm\classification-pytorch-main\logs'
    target_path = r'D:\APPDATAiD\aliyunpan\vit\\'
    ele = os.listdir(move_path)
    elem = ele[::-1][0]
    soe_path = os.path.join(move_path, elem)
    shutil.move(soe_path, target_path)

    from osgeo import gdal
    from classification import Classification
    import os
    import numpy as np

    mean1 = 0.29842865019998754
    mean2 = 0.29291234947708983
    mean3 = 0.1590252907804978
    mean4 = 0.34087137754443153
    mean5 = 0.7678303623294395
    mean6 = 0.20349380596780828
    std1 = 0.14394715026981078
    std2 = 0.13205599131085333
    std3 = 0.09776519023811078
    std4 = 0.10343500177333416
    std5 = 0.144600133615306
    std6 = 0.06519548892734287


    def Pression(path):
        pic_path = path  # 需要修改的图片路径
        pic_name = os.listdir(pic_path)  # 获取原路径下的图片
        img_num = len(pic_name)
        pre_false = 0
        for i in range(len(pic_name)):
            name = pic_name.pop()
            try:
                image = gdal.Open(pic_path + name)
                image = image.ReadAsArray()
                image = image.transpose(1, 2, 0)
                image = np.array(image).astype(np.float32)
                # image = np.nan_to_num(image)
                image[np.isnan(image)] = 0
                # ---------------------正则化数据START---------------------------
                # 通道修改处
                image[image < 0] = 0

                position1 = np.where(image[:, :, 0] == 0)
                position2 = np.where(image[:, :, 1] == 0)
                position3 = np.where(image[:, :, 2] == 0)
                position4 = np.where(image[:, :, 3] == 0)
                position5 = np.where(image[:, :, 4] == 0)
                position6 = np.where(image[:, :, 5] == 0)
                # position7 = np.where(image[:, :, 6] == 0)
                # position8 = np.where(image[:, :, 7] == 0)
                #
                ch1 = image[:, :, 0]
                ch1[position1] = mean1
                image[:, :, 0] = ch1
                image[:, :, 0] = (image[:, :, 0] - mean1) / std1

                ch2 = image[:, :, 1]
                ch2[position2] = mean2
                image[:, :, 1] = ch2
                image[:, :, 1] = (image[:, :, 1] - mean2) / std2

                ch3 = image[:, :, 2]
                ch3[position3] = mean3
                image[:, :, 2] = ch3
                image[:, :, 2] = (image[:, :, 2] - mean3) / std3

                ch4 = image[:, :, 3]
                ch4[position4] = mean4
                image[:, :, 3] = ch4
                image[:, :, 3] = (image[:, :, 3] - mean4) / std4

                ch5 = image[:, :, 4]
                ch5[position5] = mean5
                image[:, :, 4] = ch5
                image[:, :, 4] = (image[:, :, 4] - mean5) / std5

                ch6 = image[:, :, 5]
                ch6[position6] = mean6
                image[:, :, 5] = ch6
                image[:, :, 5] = (image[:, :, 5] - mean6) / std6

                # ch7 = image[:, :, 6]
                # ch7[position7] = mean7
                # image[:, :, 6] = ch7
                # image[:, :, 6] = (image[:, :, 6] - mean7) / std7
                #
                # ch8 = image[:, :, 7]
                # ch8[position8] = mean8
                # image[:, :, 7] = ch8
                # image[:, :, 7] = (image[:, :, 7] - mean8) / std8
                # ---------------------正则化数据END---------------------------

            except:
                print('Open Error! Try again!')
                continue
            else:

                class_name = classfication.detect_image(image)
                class_ture = pic_path.split('/')[2]
                if class_name != class_ture:
                    pre_false = pre_false + 1
        ture_rate = 1 - pre_false / img_num
        false_rate = pre_false / img_num

        return pre_false, img_num, ture_rate, false_rate


    path1 = 'datasets/test/0%/'
    path2 = 'datasets/test/10%~20%/'
    path3 = 'datasets/test/50%~60%/'
    path4 = 'datasets/test/over_80%/'

    pic_path = r'E:\STUDYCONTENT\Pycharm\classification-pytorch-main\logs/'
    pic_name = os.listdir(pic_path)
    tem1 = 0
    tem2 = 0
    best_top1score = 0
    best_top1model = ''
    best_avescore = 0
    best_avemodel = ''
    for i in range(len(pic_name)):
        classfication = Classification(i)
        pre_false1, img_num1, ture_rate1, false_rate1 = Pression(path1)
        pre_false2, img_num2, ture_rate2, false_rate2 = Pression(path2)
        pre_false3, img_num3, ture_rate3, false_rate3 = Pression(path3)
        pre_false4, img_num4, ture_rate4, false_rate4 = Pression(path4)
        if 1 - (pre_false1 + pre_false2 + pre_false3 + pre_false4) / (img_num1 + img_num2 + img_num3 + img_num4) > tem1:
            tem1 = 1 - (pre_false1 + pre_false2 + pre_false3 + pre_false4) / (img_num1 + img_num2 + img_num3 + img_num4)
            best_top1score = tem1
            best_top1model = pic_name[i]
        if (ture_rate1 + ture_rate2 + ture_rate3 + ture_rate4) / 4 > tem2:
            tem2 = (ture_rate1 + ture_rate2 + ture_rate3 + ture_rate4) / 4
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
        print("-" * 10 + "总预测情况" + "-" * 10)
        print('预测准确率：', str((1 - (pre_false1 + pre_false2 + pre_false3 + pre_false4) / (
                img_num1 + img_num2 + img_num3 + img_num4)) * 100) + '%')
        print('四类平均准确率：', str(((ture_rate1 + ture_rate2 + ture_rate3 + ture_rate4) / 4) * 100) + '%')

    # print('\033[1;31;42m')
    # print("-" * 10 + "最好的模型" + "-" * 10)
    # print('top1最高分：', str(best_top1score * 100) + '%')
    # print('top1最好的模型：', best_top1model)a
    # print('ave最高分：', str(best_avescore * 100) + '%')
    # print('ave最好的模型：', best_avemodel)
    # print('\033[0m')

    print('\033[1;31;42m')
    print("-" * 10 + "最好的模型" + "-" * 10)
    print('top1最高分：', str(best_top1score * 100) + '%')
    str_res = 'top1最高分：', str(best_top1score * 100) + '%'
    print('top1最好的模型：', best_top1model)
    str_res += 'top1最好的模型：', best_top1model
    print('ave最高分：', str(best_avescore * 100) + '%')
    str_res += 'ave最高分：', str(best_avescore * 100) + '%'
    print('ave最好的模型：', best_avemodel)
    str_res += 'ave最好的模型：', best_avemodel
    with open('res.txt', 'w') as file_res:
        file_res.write(str(str_res))
    file_res.close()
    print('\033[0m')