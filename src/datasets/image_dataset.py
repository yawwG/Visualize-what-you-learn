import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
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
        self.masktransform = transforms.Compose(
            [transforms.ToTensor()])
        self.split = split

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path):

        x = cv2.imread(str(img_path), 0)
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def readmask_from_jpg(self, mask_path):

        x = cv2.imread(str(mask_path), 0)
        x = self._resize_img(x, self.cfg.data.image.imsize)
        mask = Image.fromarray(x).convert("L")

        if self.masktransform is not None:
            mask = self.masktransform(mask)

        return mask

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
        self.cfg = cfg

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(INB_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(INB_VALID_CSV)
        else:
            self.df = pd.read_csv(INB_TEST_CSV)

        # sample data
        if cfg.data.frac != 1 and split == "train":
            if self.cfg.phase == "segmentation":
                if cfg.data.frac ==0.1:
                    self.df = pd.read_csv("path")
                if cfg.data.frac ==0.01:
                    self.df = pd.read_csv('path')
            else:
                self.df = self.df.sample(frac=cfg.data.frac)
        # fill na with 0s
        self.df = self.df.fillna(0)

        super(INBImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # get image
        img_path = row["File_path"]
        img = []
        y = int(row["pathology"])

        img_path2 = []
        lcc = img_path + "_L_CC.png"
        lmlo =  img_path + "_L_MLO.png"
        rcc =  img_path + "_R_CC.png"
        rmlo =  img_path + "_R_MLO.png"
        img_path2.append(lcc)
        img_path2.append(lmlo)
        img_path2.append(rcc)
        img_path2.append(rmlo)
        for i in range(len(img_path2)):
            if (os.path.exists(img_path2[i])):
                img_tmp = cv2.imread(img_path2[i], 0)
                img_tmp = self._resize_img(img_tmp, self.cfg.data.image.imsize)
                img_tmp = Image.fromarray(img_tmp).convert("RGB")
            else:
                img_tmp = Image.new('L', (512, 512), (0)).convert("RGB")
            if self.transform is not None:
                img_tmp = self.transform(img_tmp)
                img.append(img_tmp)
        x = torch.stack(img)

        mask = []
        if self.cfg.phase == "segmentation":
            mask_path = []
            lcc_mask = ''
            lmlo_mask = ''
            rcc_mask = ''
            rmlo_mask = ''
            mask_path.append(lcc_mask)
            mask_path.append(lmlo_mask)
            mask_path.append(rcc_mask)
            mask_path.append(rmlo_mask)
            for i in range(len(mask_path)):
                if (os.path.exists(mask_path[i])):
                    mask_tmp = cv2.imread(mask_path[i], 0)
                    mask_tmp = self._resize_img(mask_tmp, self.cfg.data.image.imsize)
                    mask_tmp = Image.fromarray(mask_tmp)
                else:
                    mask_tmp = Image.new('L', (512, 512), (0))
                if self.masktransform is not None:
                    mask_tmp = self.masktransform(mask_tmp)
                mask.append(mask_tmp)
            mask = torch.stack(mask)

            y = torch.tensor(y)
            y1 = []
            for i in range(4):
                y1.append(y)
            y1 = torch.stack(y1)
            y1 = y1.squeeze()
            return  x, y1, img_path, mask
        else:
            return x, y

    def __len__(self):
        return len(self.df)

