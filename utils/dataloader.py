from random import shuffle
from osgeo import gdal
import cv2
import numpy as np
import torch.utils.data as data

from .utils import cvtColor, preprocess_input, flip180, flip90_left, flip90_right




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


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.random = random

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = gdal.Open(annotation_path)
        image = image.ReadAsArray()
        image = image.transpose(1, 2, 0)
        image = np.array(image).astype(np.float32)
        # image = np.nan_to_num(image)
        image[np.isnan(image)] = 0
        # ---------------------正则化数据START---------------------------
        # 通道修改處
        image[image < 0] = 0

        position1 = np.where(image[:, :, 0] == 0)
        position2 = np.where(image[:, :, 1] == 0)
        position3 = np.where(image[:, :, 2] == 0)
        position4 = np.where(image[:, :, 3] == 0)
        position5 = np.where(image[:, :, 4] == 0)
        position6 = np.where(image[:, :, 5] == 0)
        # position7 = np.where(image[:, :, 6] == 0)
        # position8 = np.where(image[:, :, 7] == 0)

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

        image = self.get_random_data(image, self.input_shape, random=self.random)
        image = image.transpose(2, 0, 1)
        # image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        y = int(self.annotation_lines[index].split(';')[0])
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        # image   = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        ih, iw = image[:, :, 1].shape
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = cv2.resize(image, (nw, nh))
            new_image = np.ones([w, h, 6]) * (128 / 2560)  # 通道修改处
            new_image[dy:nh + dy, dx:nw + dx] = new_image[dy:nh + dy, dx:nw + dx] + image
            image_data = new_image

            return image_data

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.75, 1.25)
        a, b = input_shape
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        if nh > a:
            nh = int(a / np.random.uniform(1, 10))
        if nw > b:
            nw = int(b / np.random.uniform(1, 10))
        # if new_ar < 1:
        #     nh = np.random.randint(1, int(h/2)+1)
        #     nw = np.random.randint(1, int(w/2)+1)
        # else:
        #     nh = np.random.randint(1, int(h/3)+1)
        #     nw = np.random.randint(1, int(w/3)+1)
        image = cv2.resize(image, (nw, nh))

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = np.ones([w, h, 6]) * (128 / 2560)  # 通道修改处
        new_image[dy:nh + dy, dx:nw + dx] = new_image[dy:nh + dy, dx:nw + dx] + image
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image[::-1]

        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-1, 2)
            if angle == -1:
                image = flip180(image)
                image = image.transpose(0, 1, 2)
            if angle == 0:
                image = flip90_left(image)
                image = image.transpose(2, 1, 0)
            if angle == 1:
                image = flip90_right(image)
                image = image.transpose(2, 1, 0)
        image_data = image
        return image_data


def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images = np.array(images)
    targets = np.array(targets)
    return images, targets
