import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class LEGOdataset(Dataset):
    def __init__(self,
                 split='train',
                 augumentation=None,
                 preprocessing=None,
                 train=True,
                 train_fraction=0.8):

        self.augumentation = augumentation
        self.split = split
        self.preprocessing = preprocessing

        if not os.path.exists(split + '_images'):
            os.system("!wget https://www.dropbox.com/s/njlvgfplr34vyu8/dmia-dl-aut19-segmentation.zip -q")
            os.system("!unzip dmia-dl-aut19-segmentation.zip")

        path = '/content/{}_images/{}_images'.format(self.split, self.split)
        self.images = os.listdir(path)

        ids = list(range(len(self.images)))
        random.shuffle(ids)
        train_size = int(len(ids) * train_fraction)
        self.indices = ids[:train_size] if train else ids[train_size:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_name = self.images[self.indices[index]]
        path = '/content/{}_images/{}_images'.format(self.split, self.split)
        img_path = os.path.join(path, img_name, 'images', img_name + '.png')
        mask_path = os.path.join(path, img_name, 'masks')
        img = np.array(Image.open(img_path))

        if self.split == 'test':
            img = torch.tensor(img).permute(2, 0, 1)
            return {'image': img.float() / 255.0}

        if self.split == 'train':
            mask_objs = os.listdir(mask_path)
            mask = np.zeros((img.shape[0], img.shape[1]))
            for obj in mask_objs:
                mask += np.array(Image.open(os.path.join(mask_path, obj)))

            if self.preprocessing is not None:
                res = self.preprocessing(image=img, mask=mask)
                img, mask = res['image'], res['mask']

            if self.augumentation is not None:
                aug = self.augumentation(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']

            return {'image': img.float(), 'mask': mask.squeeze(0).long() // 255}





