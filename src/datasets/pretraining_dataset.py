import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from VSWL.src.constants import *
class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor()])
        csv_path = os.path.join(INB_DATA_DIR, INB_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # load studies and study to text mapping
        self.filenames, self.path2sent, self.label, self.ato, self.keyword = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):

        filepath = os.path.join("captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[INB_SPLIT_COL] == split][
            INB_PATH_COL
        ].tolist()
        label = self.df[self.df[INB_SPLIT_COL] == split][
            'pathology'].tolist()
        ato = self.df[self.df[INB_SPLIT_COL] == split][
            'ato'
        ].tolist()

        keyword = self.df[self.df[INB_SPLIT_COL] == split][
            'keyword'
        ].tolist()

        return filenames, path2sent, label, ato, keyword

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        lcc =  "path"
        lmlo = "path"
        rcc = "path"
        rmlo = "path"
        x1 = cv2.imread(lcc, 0)
        x2 = cv2.imread(lmlo, 0)
        x3 = cv2.imread(rcc, 0)
        x4 = cv2.imread(rmlo, 0)
        mask_path =[]
        lcc_mask = "path"
        lmlo_mask = "path"
        rcc_mask = "path"
        rmlo_mask = "path"
        mask_path.append(lcc_mask)
        mask_path.append(lmlo_mask)
        mask_path.append(rcc_mask)
        mask_path.append(rmlo_mask)

        mask = []
        for i in range(len(mask_path)):
            if (os.path.exists(mask_path[i])):
                mask_tmp = cv2.imread(mask_path[i], 0)
                mask_tmp = self._resize_img(mask_tmp, self.cfg.data.image.imsize)
                mask_tmp = Image.fromarray(mask_tmp)
            else:
                mask_tmp = Image.new('L', (512, 512), (0))

            mask_tmp = self.tensor_transform(mask_tmp)
            mask.append(mask_tmp)

        mask = torch.stack(mask)

        # tranform images
        x1 = self._resize_img(x1, self.cfg.data.image.imsize)
        img1 = Image.fromarray(x1).convert("RGB")
        x2 = self._resize_img(x2, self.cfg.data.image.imsize)
        img2 = Image.fromarray(x2).convert("RGB")
        x3 = self._resize_img(x3, self.cfg.data.image.imsize)
        img3 = Image.fromarray(x3).convert("RGB")
        x4 = self._resize_img(x4, self.cfg.data.image.imsize)
        img4 = Image.fromarray(x4).convert("RGB")

        if transform is not None:
            img1 = transform(img1)
            img2 = transform(img2)
            img3 = transform(img3)
            img4 = transform(img4)
            img = torch.cat((img1,img2,img3,img4),0)

        return img, mask #[3*4,w,h]

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs,mask = self.get_imgs(key, self.transform)
        ato = self.ato[index]

        xx = ato.split()
        xx = np.array(xx).astype(float)
        ato = torch.FloatTensor(xx.reshape((3, 1, 1)))

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)

        keyword = self.keyword[index]

        return imgs, mask, caps, cap_len, key, label, ato, keyword

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        # translator = Translator()
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[INB_REPORT_COL]) == str:

                captions += row[INB_REPORT_COL]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    # if len(t) > 0:
                    included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))
            path2sent[row[INB_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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

def multimodal_collate_fn(batch):
    """sort sequence"""

    imgs, imgslcc, imgslmlo, imgsrcc, imgsrmlo, masks, cap_len, ids, tokens, attention, path, labels, atomo, keywords =[], [], [], [], [], [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, msk, cap, cap_l, p, label, ato, keyword = b
        img1 = img.view(4, 3, img.size(1), img.size(2))
        imgslcc.append(torch.split(img1, 1, dim=0)[0])
        # imgslcc = imgslcc.view(imgslcc.size(0))
        imgslmlo.append(torch.split(img1, 1, dim=0)[1])
        imgsrcc.append(torch.split(img1, 1, dim=0)[2])
        imgsrmlo.append(torch.split(img1, 1, dim=0)[3])
        imgs.append(img)
        masks.append(msk)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        # path.append(p.split("/")[-1].split("_")[0])
        path.append(p)
        labels.append(label)
        atomo.append(ato)
        keywords.append(keyword)

    # stack
    imgs = torch.stack(imgs)
    imgslcc = torch.stack(imgslcc)[:, 0, :, :, :]
    imgslmlo = torch.stack(imgslmlo)[:, 0, :, :, :]
    imgsrcc = torch.stack(imgsrcc)[:, 0, :, :, :]
    imgsrmlo = torch.stack(imgsrmlo)[:, 0, :, :, :]
    masks = torch.stack(masks)
    # ids = torch.repeat(ids, 4)
    # ids = [itm for itm in ids for i in range(4)]
    ids = torch.stack(ids).squeeze()
    # tokens = [itm for itm in tokens for i in range(4)]
    tokens = torch.stack(tokens).squeeze()
    # attention = [itm for itm in attention for i in range(4)]
    attention = torch.stack(attention).squeeze()
    # labels = [itm for itm in labels for i in range(4)]
    label = torch.stack(labels)
    # atomo = [itm for itm in atomo for i in range(4)]
    atomo = torch.stack(atomo)
    # keyword = torch.stack(keywords)
    # keyword = keywords

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    path_tmp = []
    keyword_tmp = []
    for i in range(len(path)):
        lcc = "path"
        lmlo = "path"
        rcc = "path"
        rmlo = "path"
        path_tmp.append(lcc)
        path_tmp.append(lmlo)
        path_tmp.append(rcc)
        path_tmp.append(rmlo)
        keyword_tmp.append(keywords[sorted_cap_indices[i]])

    return_dict = {
        "imgs": imgs[sorted_cap_indices].view(len(path) * 4, 3, imgs[sorted_cap_indices].size(2),
                                              imgs[sorted_cap_indices].size(3)),
        "caption_ids": torch.stack([itm for itm in ids[sorted_cap_indices] for i in range(4)]),
        "token_type_ids": torch.stack([itm for itm in tokens[sorted_cap_indices] for i in range(4)]),
        "attention_mask": torch.stack([itm for itm in attention[sorted_cap_indices] for i in range(4)]),
        "imgslcc": imgslcc[sorted_cap_indices].view(len(path), 3, imgslcc[sorted_cap_indices].size(2),
                                                    imgslcc[sorted_cap_indices].size(3)),
        "imgslmlo": imgslmlo[sorted_cap_indices].view(len(path), 3, imgslmlo[sorted_cap_indices].size(2),
                                                      imgslmlo[sorted_cap_indices].size(3)),
        "imgsrcc": imgsrcc[sorted_cap_indices].view(len(path), 3, imgsrcc[sorted_cap_indices].size(2),
                                                    imgsrcc[sorted_cap_indices].size(3)),
        "imgsrmlo": imgsrmlo[sorted_cap_indices].view(len(path), 3, imgsrmlo[sorted_cap_indices].size(2),
                                                      imgsrmlo[sorted_cap_indices].size(3)),
        "mask": masks[sorted_cap_indices].view(len(path) * 4, 1, masks[sorted_cap_indices].size(3),
                                               masks[sorted_cap_indices].size(4)),
        "cap_lens": [itm for itm in sorted_cap_lens for i in range(4)],
        # "path": [itm for itm in path_tmp for i in range(4)],
        "path": path_tmp,
        "label": torch.stack([itm for itm in label[sorted_cap_indices] for i in range(4)]),
        "ato": torch.stack([itm for itm in atomo[sorted_cap_indices] for i in range(4)]),
        "keyword": keyword_tmp
    }

    return return_dict