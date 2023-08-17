import numpy as np
import torch
from torch import nn
import os
from nets import get_model_from_name
from utils.utils import ( get_classes, letterbox_image)

class Classification(object):
        _defaults = {
            "classes_path"  : 'model_data/cls_classes.txt',

            "input_shape"   : [224, 224],

            "backbone"      : 'vit',

            "cuda"          : True
        }

        @classmethod
        def get_defaults(cls, n):
            if n in cls._defaults:
                return cls._defaults[n]
            else:
                return "Unrecognized attribute name '" + n + "'"

        def __init__(self,n, **kwargs):
            pic_path = 'E:\STUDYCONTENT\Pycharm\classification-pytorch-main\logs/'
            pic_name = os.listdir(pic_path)
            self.n = n
            self.model_path = pic_path+pic_name[self.n]
            self.__dict__.update(self._defaults)
            for name, value in kwargs.items():
                setattr(self, name, value)
            self.class_names, self.num_classes = get_classes(self.classes_path)
            self.generate()
        def generate(self):
            if self.backbone != "vit":
                self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
            else:
                self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
            device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.model  = self.model.eval()
            print('{} model, and classes loaded.'.format(self.model_path))

            if self.cuda:
                self.model = nn.DataParallel(self.model)
                self.model = self.model.cuda()
        def detect_image(self, image):
            image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
            image_data  = np.transpose(np.expand_dims(np.array(image_data, np.float32), 0), (0, 3, 1, 2))

            with torch.no_grad():
                photo   = torch.from_numpy(image_data)
                if self.cuda:
                    photo = photo.cuda()
                preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
            class_name  = self.class_names[np.argmax(preds)]
            probability = np.max(preds)

            return class_name
