from random import shuffle
from osgeo import gdal
import cv2
import numpy as np
import torch.utils.data as data


from .utils import cvtColor, preprocess_input ,flip180,flip90_left,flip90_right


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True):
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.random             = random

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = cv2.imread(annotation_path,-1)
        # image = image.ReadAsArray()
        # image = image.transpose(1, 2, 0)
        image = np.array(image).astype(np.float32)
        image = (np.nan_to_num(image))/255
        image = self.get_random_data(image, self.input_shape, random=self.random)
        image = (image.transpose(2, 0, 1))
        # image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        y = int(self.annotation_lines[index].split(';')[0])
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        # image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        ih, iw = image[:, :, 1].shape
        h, w = input_shape


        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image = cv2.resize(image, (nw, nh))
            new_image = np.ones([w, h, 3]) * (128 / 2560)
            new_image[dy:nh + dy, dx:nw + dx] = new_image[dy:nh + dy, dx:nw + dx] + image
            image_data = new_image

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
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

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = np.ones([w, h, 3]) * (128 / 2560)
        new_image[dy:nh + dy, dx:nw + dx] = new_image[dy:nh + dy, dx:nw + dx] + image
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
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
