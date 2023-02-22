import os
import random
from copy import copy
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class ImageNetCore(Dataset):
    
    def __init__(self, path, split, transform=None):
        super().__init__()
        assert split in ('train', 'val', 'test')
        self.split = split
        self.path = path
        self.transform = transform
        self.imgs = []

        self.load()


    def load(self):

        def _enumerate_image():
            imgsDir = self.path + '/' + self.split
            imgList = os.listdir(imgsDir)
            for j, imgName in enumerate(imgList):
                imgPath = self.path + '/' + self.split + '/' + imgName
                img = Image.open(imgPath)
                transform = transforms.Compose([
                    transforms.PILToTensor()
                ])
                img_tensor = transform(img)
                yield img_tensor, j
        
        for img, img_id in _enumerate_image():
            self.imgs.append({
                'image': img,
                'id': img_id,
            })

            

        # Deterministically shuffle the dataset
        self.imgs.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.imgs[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

