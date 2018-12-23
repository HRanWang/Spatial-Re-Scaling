from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import random
import math

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None,random_mask=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.random_mask = random_mask
        if self.random_mask:
            self.random_mask_obj = RandomErasing(random_fill=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.random_mask:
            img = self.random_mask_obj(img)
        return img, fname, pid, camid

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    random_fill: If ture, fill the erased area with random number. If false: fill with image net mean.
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=(0., 0., 0.), random_fill=False):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.random_fill=random_fill

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if not self.random_fill:
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                else:
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                        img[1, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                        img[2, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = torch.rand((h, w))
                return img

        return img
