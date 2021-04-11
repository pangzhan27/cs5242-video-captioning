import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import os, pickle, json
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,

                                                                                                interpolation=self.interpolation)

class ChangeColor(object):
    def __init__(self, hue, brightness, contrast):
        self.hue = hue
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        out = F.adjust_brightness(img, self.brightness)
        out = F.adjust_contrast(out, self.contrast)
        out = F.adjust_hue(out, self.hue)
        return out

class VID2016(data.Dataset):
    def __init__(self, ds, phase='train', resize = 224):
        self.ds = ds
        self.phase = phase
        self.transform = None
        self.inp = []
        ### data augmentation
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=self.normalization_mean,std=self.normalization_std)
        if phase == 'train':
            self.transform = transforms.Compose([
                Warp(resize+12),
                transforms.RandomCrop(resize),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.ToTensor(),
                normalize,
            ])
        elif phase == 'test':
            self.transform = transforms.Compose([
                Warp(resize),
                ChangeColor(-0.25, 0.75, 0.75),  # ChangeColor(0.25,1.25,1.25),
                transforms.ToTensor(),
                normalize,
            ])
        ## read data
        self.get_anno()
        self.num_obj = len(self.obj2idx)
        self.num_rel = len(self.rel2idx)
        self.img_list = sorted(list(os.listdir(os.path.join(self.ds, self.phase))))



    def get_anno(self):
        self.obj2idx = json.load(open(os.path.join(self.ds, 'object1_object2.json'), 'r'))
        self.rel2idx = json.load(open(os.path.join(self.ds, 'relationship.json'), 'r'))
        if self.phase == 'train':
            self.img2target = json.load(open(os.path.join(self.ds, 'training_annotation.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        video_id = self.img_list[index]
        temp_path = os.path.join(self.ds, self.phase, video_id)
        img_ids = sorted(list(os.listdir(temp_path)))
        if self.phase == 'train':
            offset = np.array(list(range(0, 30, 3)))
            index = np.random.randint(0, 3, size=1)[0]
            used_id = index + offset
            used_img_ids = [img_ids[i] for i in used_id]
            images = torch.stack(list(map(lambda x: self.get(temp_path, x), used_img_ids)))
        else:
            offset = np.array(list(range(0, 30, 3)))
            images = []
            for i in range(3):
                used_id = i + offset
                used_img_ids = [img_ids[i] for i in used_id]
                images.append(torch.stack(list(map(lambda x: self.get(temp_path, x), used_img_ids))))
            images = torch.cat(images, dim=0)

        obj1, rel, obj2 = -1, -1, -1
        if self.phase == 'train':
            obj1, rel, obj2 = self.img2target[video_id]
        return (images, video_id), (obj1, rel, obj2)

    def get(self, path, name):
        img = Image.open(os.path.join(path, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
