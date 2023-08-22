import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from VSWL.src.constants import *
from torchvision import transforms

class ImageBaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        transform=None,
    ):

        self.cfg = cfg
        self.transform = transform
        self.split = split

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path):

        x = cv2.imread(str(img_path), 0)
        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def read_from_dicom(self, img_path):
        raise NotImplementedError

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
class INBImageDataset(ImageBaseDataset):
    def __init__(self, cfg, split="train", transform=None):

        print('load cmmd dataset!')
        self.cfg = cfg
        self.df = pd.read_csv('')
        self.df = self.df[self.df[INB_SPLIT_COL] == split]
        # sample data
        if cfg.data.frac != 1 and split == "train":
            self.df = self.df.sample(frac=cfg.data.frac)
        self.masktransform = transforms.Compose(
            [transforms.ToTensor()])
        self.root_dir ='path'
        self.mask_dir = 'path'
        self.images = os.listdir(self.root_dir)
        # fill na with 0s
        self.df = self.df.fillna(0)

        super(INBImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # get image
        image_index = row["filename"]
        img_path = os.path.join(self.root_dir, image_index+'.png')
        mask_path = os.path.join(self.mask_dir, image_index.split('_')[0] +'_mask.png')
        x = self.read_from_jpg(img_path)
        # get labels
        y = int(row["pathology"])
        y = torch.tensor(y)
        if self.cfg.phase == "segmentation":
            if (os.path.exists(mask_path) and y != 0):
                mask = cv2.imread(mask_path, 0)
                mask = self._resize_img(mask, self.cfg.data.image.imsize)
                mask = Image.fromarray(mask)
                segment_label=1
            else:
                mask = Image.new('L', (512, 512), (0))
                segment_label = 0
            if self.transform is not None:
                mask = self.masktransform(mask)
            mask = torch.squeeze(mask)
            return x, mask, mask_path, segment_label

        else:
            return x, y

    def __len__(self):
        return len(self.df)

