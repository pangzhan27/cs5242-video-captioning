import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import os, pickle, json
import torchvision.transforms as transforms
import torch

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__

class VID2016(data.Dataset):
    def __init__(self, ds, phase='train', resize = 224, inp_name='adj/embedding.pickle'):
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(resize),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.ToTensor(),
                normalize,
            ])
        elif phase == 'test':
            self.transform = transforms.Compose([
                Warp(resize),
                transforms.ToTensor(),
                normalize,
            ])
        ## read data
        self.get_anno()
        self.num_obj = len(self.obj2idx)
        self.num_rel = len(self.rel2idx)
        self.img_list = sorted(list(os.listdir(os.path.join(self.ds, self.phase))))

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        a=1


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
        target = tuple()
        if self.phase == 'train':
            obj_target = np.zeros(self.num_obj, np.float32)
            rel_target = np.ones(1, np.int)
            obj1, rel, obj2 = self.img2target[video_id]
            obj_target[obj1] = 1
            obj_target[obj2] = 1
            rel_target = rel_target * rel
            target = (obj_target, rel_target)
        return (images, video_id, self.inp), target

    def get(self, path, name):
        img = Image.open(os.path.join(path, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
