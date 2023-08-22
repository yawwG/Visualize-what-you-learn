import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics

from PIL import Image
from .. import builder
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer


class VSWL(nn.Module):
    def __init__(self, cfg):
        super(VSWL, self).__init__()

        self.cfg = cfg

        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder_cc = builder.build_img_model(cfg)
        self.img_encoder_mlo = builder.build_img_model(cfg)

        self.img_decoder = builder.build_img_decoder(cfg)
        self.dh_loss = VSWL.src.loss.VSWL_loss.dh_loss
        self.blur_loss = VSWL.src.loss.VSWL_loss.Blur_loss

        self.adv_g_loss = VSWL.src.loss.VSWL_loss.inner_loss_g
        self.adv_d_loss = VSWL.src.loss.VSWL_loss.inner_loss_d

        self.local_loss = VSWL.src.loss.VSWL_loss.local_loss
        self.global_loss = VSWL.src.loss.VSWL_loss.global_loss
        self.local_loss_weight = self.cfg.model.VSWL.local_loss_weight
        self.global_loss_weight = self.cfg.model.VSWL.global_loss_weight

        self.temp1 = self.cfg.model.VSWL.temp1
        self.temp2 = self.cfg.model.VSWL.temp2
        self.temp3 = self.cfg.model.VSWL.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2 = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents,text_mlp1, text_mlp2

    def image_encoder_forward(self, imgslcc, imgslmlo, imgsrcc, imgsrmlo):

        img_feat_g1, img_emb_l1, x_1, e11, e21, e31 = self.img_encoder_cc(imgslcc, get_local=True)
        img_feat_g2, img_emb_l2, x_2, e12, e22, e32 = self.img_encoder_mlo(imgslmlo, get_local=True)
        img_feat_g3, img_emb_l3, x_3, e13, e23, e33 = self.img_encoder_cc(imgsrcc, get_local=True)
        img_feat_g4, img_emb_l4, x_4, e14, e24, e34 = self.img_encoder_mlo(imgsrmlo, get_local=True)

        img_feat_g = torch.cat((img_feat_g1, img_feat_g2, img_feat_g3, img_feat_g4),0)
        img_emb_l = torch.cat((img_emb_l1, img_emb_l2, img_emb_l3, img_emb_l4),0)

        img_emb_g, img_emb_l = self.img_encoder_cc.generate_embeddings(
            img_feat_g,
            img_emb_l
        )  # [b,768,w,h]
        img_emb_g_new = torch.cat((torch.split(img_emb_g,1,dim=0)[0], torch.split(img_emb_g,1,dim=0)[3], torch.split(img_emb_g,1,dim=0)[6], torch.split(img_emb_g,1,dim=0)[9],
                                   torch.split(img_emb_g,1,dim=0)[1], torch.split(img_emb_g,1,dim=0)[4], torch.split(img_emb_g,1,dim=0)[7], torch.split(img_emb_g,1,dim=0)[10],
                                   torch.split(img_emb_g,1,dim=0)[2], torch.split(img_emb_g,1,dim=0)[5], torch.split(img_emb_g,1,dim=0)[8], torch.split(img_emb_g,1,dim=0)[11]), 0)
        img_emb_l_new = torch.cat((torch.split(img_emb_l,1,dim=0)[0], torch.split(img_emb_l,1,dim=0)[3], torch.split(img_emb_l,1,dim=0)[6], torch.split(img_emb_l,1,dim=0)[9],
                                   torch.split(img_emb_l,1,dim=0)[1], torch.split(img_emb_l,1,dim=0)[4], torch.split(img_emb_l,1,dim=0)[7], torch.split(img_emb_l,1,dim=0)[10],
                                   torch.split(img_emb_l,1,dim=0)[2], torch.split(img_emb_l,1,dim=0)[5], torch.split(img_emb_l,1,dim=0)[8], torch.split(img_emb_l,1,dim=0)[11]), 0)

        x_ = torch.cat((torch.split(x_1, 1, dim=0)[0], torch.split(x_2, 1, dim=0)[0], torch.split(x_3, 1, dim=0)[0], torch.split(x_4, 1, dim=0)[0],
                                   torch.split(x_1, 1, dim=0)[1], torch.split(x_2, 1, dim=0)[1], torch.split(x_3, 1, dim=0)[1], torch.split(x_4, 1, dim=0)[1],
                                   torch.split(x_1, 1, dim=0)[2], torch.split(x_2, 1, dim=0)[2], torch.split(x_3, 1, dim=0)[2], torch.split(x_4, 1, dim=0)[2]), 0)
        e1 = torch.cat((torch.split(e11, 1, dim=0)[0], torch.split(e12, 1, dim=0)[0], torch.split(e13, 1, dim=0)[0],
                        torch.split(e14, 1, dim=0)[0],
                        torch.split(e11, 1, dim=0)[1], torch.split(e12, 1, dim=0)[1], torch.split(e13, 1, dim=0)[1],
                        torch.split(e14, 1, dim=0)[1],
                        torch.split(e11, 1, dim=0)[2], torch.split(e12, 1, dim=0)[2], torch.split(e13, 1, dim=0)[2],
                        torch.split(e14, 1, dim=0)[2]), 0)
        e2 = torch.cat((torch.split(e21, 1, dim=0)[0], torch.split(e22, 1, dim=0)[0], torch.split(e23, 1, dim=0)[0],
                        torch.split(e24, 1, dim=0)[0],
                        torch.split(e21, 1, dim=0)[1], torch.split(e22, 1, dim=0)[1], torch.split(e23, 1, dim=0)[1],
                        torch.split(e24, 1, dim=0)[1],
                        torch.split(e21, 1, dim=0)[2], torch.split(e22, 1, dim=0)[2], torch.split(e23, 1, dim=0)[2],
                        torch.split(e24, 1, dim=0)[2]), 0)
        e3 = torch.cat((torch.split(e31, 1, dim=0)[0], torch.split(e32, 1, dim=0)[0], torch.split(e33, 1, dim=0)[0],
                        torch.split(e34, 1, dim=0)[0],
                        torch.split(e31, 1, dim=0)[1], torch.split(e32, 1, dim=0)[1], torch.split(e33, 1, dim=0)[1],
                        torch.split(e34, 1, dim=0)[1],
                        torch.split(e31, 1, dim=0)[2], torch.split(e32, 1, dim=0)[2], torch.split(e33, 1, dim=0)[2],
                        torch.split(e34, 1, dim=0)[2]), 0)

        return img_emb_l_new, img_emb_g_new, x_, e1, e2, e3

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents, label,keyword,fc):
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps, l_coarse_to_fine_loss = self.local_loss(
            img_emb_l,
            text_emb_l,

            cap_lens,
            label,
            keyword,
            fc,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps, l_coarse_to_fine_loss

    def _calc_global_loss(self, img_emb_g, text_emb_g, keyword, fc):
        g_loss0, g_loss1, g_coarse_to_fine_loss = self.global_loss(img_emb_g, text_emb_g, keyword, fc, temp3=self.temp3)
        return g_loss0, g_loss1, g_coarse_to_fine_loss

    def calc_loss(self, img_emb_l, img_emb_g, m, y1, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, x, current_epoch):
        fc = nn.Linear(img_emb_g.size(1) * img_emb_g.size(0) * 2, img_emb_g.size(0) * img_emb_g.size(0) * 4).cuda()

        l_loss0, l_loss1, attn_maps, l_coarse_to_fine_loss = self._calc_local_loss(
            img_emb_l, text_emb_l, sents, x['label'], x['keyword'], fc,
        )
        g_loss0, g_loss1, g_coarse_to_fine_loss = self._calc_global_loss(img_emb_g, text_emb_g, x['keyword'], fc)

        dh_loss0 = self._calc_dh_loss(m, y1, x["imgs"], x["ato"])
        dh_loss1 = self._calc_blur_m_loss(m, y1, x["imgs"])

        fc =0
        inner_g = self._calc_adv_g_loss(img_emb_g, text_mlp1, text_mlp2, x["imgs"].size(0),fc)
        inner_l = self._calc_adv_d_loss(img_emb_l, text_mlp1, text_mlp2, x["imgs"].size(0),fc)

        loss = 0
        interLoss = (g_loss0 + g_loss1) + (l_loss0 + l_loss1)
        # loss += interLoss # inter-modal loss

        coarse_to_fineLoss =  ( g_coarse_to_fine_loss +  l_coarse_to_fine_loss)
        # loss += coarse_to_fineLoss * 0.001 #coarse_to_fine loss

        recLoss = (dh_loss0 + 0.005 * dh_loss1)
        # loss += recLoss # generation loss

        innerLoss= (inner_l + inner_g) #intra-modal loss
        # loss += innerLoss * 0.001

        if self.cfg.ablation == '0': #all setting
            loss = interLoss + coarse_to_fineLoss * 0.001 + innerLoss * 0.001 + recLoss
        if self.cfg.ablation=='1':
            loss += interLoss
            loss += coarse_to_fineLoss * 0.001
            loss += innerLoss * 0.001
        elif self.cfg.ablation == '2':
            loss += coarse_to_fineLoss * 0.001
            loss += recLoss
            loss += innerLoss * 0.001
        elif self.cfg.ablation == '3':
            loss += interLoss
            loss += recLoss
            loss += innerLoss * 0.001
        elif self.cfg.ablation == '4':
            loss += interLoss
            loss += coarse_to_fineLoss * 0.001
            loss += recLoss
        elif self.cfg.ablation == '5':
            loss += interLoss
            loss += recLoss
        elif self.cfg.ablation == '6':
            loss += innerLoss
            loss += recLoss
        elif self.cfg.ablation == '7':
            loss += coarse_to_fineLoss * 0.001
            loss += recLoss

        return loss, attn_maps

    def _calc_adv_g_loss(self, i, t1, t2, b, fc):
        inner_g, adv_g_loss = self.adv_g_loss( i, t1, t2, b, fc)
        return inner_g

    def _calc_adv_d_loss(self,  i, t1, t2, b, fc):
        inner_l, adv_d_loss = self.adv_d_loss( i, t1, t2, b, fc)
        return inner_l


    def _calc_dh_loss(self, m, y, images, ato):
        dh_loss = self.dh_loss(m, y, images, ato)
        return dh_loss

    def _calc_blur_m_loss(self, m, y, images):
        blur_loss = self.blur_loss(m, y, images)
        return blur_loss

    def forward(self, x):

        # img encoder branch
        img_emb_l, img_emb_g, x_, e1, e2, e3 =  self.image_encoder_forward(x["imgslcc"],x["imgslmlo"],x["imgsrcc"],x["imgsrmlo"])

        # img decoder branch
        y1, m = self.img_decoder(x_, e1, e2, e3, True)

        # text encorder branch
        text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2 = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_l, img_emb_g, m, y1, text_emb_l, text_emb_g, sents,text_mlp1, text_mlp2

    def plot_post(self, m_, mask, imgs, ato, current_epoch, path):
        shape = [imgs.size(0), 1, m_.size(2), m_.size(3)]
        t = np.zeros(shape).astype(np.float32)
        zeros = torch.zeros(1,dtype=torch.float16).cuda()
        ones = torch.ones(1,dtype=torch.float16).cuda()
        ones_ = ones * 0.58
        # label_mask2 = torch.ones(label_mask2.shape, dtype=torch.long).cuda()
        ones_ = ones_.type(dtype=torch.float16)
        # trans_m = torch.where(m_ > ones_, zeros, m_)
        # trans_m = 1-m_
        for i in range(imgs.size(0)):
            t[i, :, :, :] = utils.t_matting(imgs.cpu().numpy()[i], m_.detach().cpu().numpy()[i])
            # t[i, :, :, :] = t_matting(images_torch.cpu().numpy()[i], mask_[i].detach().cpu().numpy())
            #
        t = torch.from_numpy(t).cuda()
        # post = (imgs - (t * ato)) / (1 - t)
        post = (imgs - ((1 - t) * ato)) / t
        mean = imgs - post
        mean = (mean + 1) / 2
        post = (post + 1) / 2
        post = torch.clip(post, 0, 1)
        imgs = (imgs + 1) / 2
        a = utils.torch_to_np(imgs)
        print("--------------:", np.max(a))
        # rr=post.detach().cpu().numpy()
        # test = np.clip(a, 0, 1)

        middlepost = str(
            path.split("/")[-1].split(".png")[0]) +str(current_epoch) + "_input_"
        utils.save_image(middlepost, a, self.cfg.output_dir)

        a = utils.torch_to_np(post)
        print("--------------:", np.max(a))
        # rr=post.detach().cpu().numpy()
        # test = np.clip(a, 0, 1)

        middlepost = str(
            path.split("/")[-1].split(".png")[0]) + str(current_epoch) + "_Post_"
        utils.save_image(middlepost, a, self.cfg.output_dir)

        a2 = utils.torch_to_np(m_)
        # rr=post.detach().cpu().numpy()
        # test = np.clip(a, 0, 1)

        middlepost =  str(
            path.split("/")[-1].split(".png")[0]) +str(current_epoch) + "_M_"
        utils.save_image(middlepost, a2, self.cfg.output_dir)

        a2 = utils.torch_to_np(1 - m_)
        # rr=post.detach().cpu().numpy()
        # test = np.clip(a, 0, 1)

        middlepost =str(
            path.split("/")[-1].split(".png")[0])+ str(current_epoch) + "_1-M_"
        utils.save_image(middlepost, a2, self.cfg.output_dir)

        a3 = utils.torch_to_np(mean)
        # rr=post.detach().cpu().numpy()
        # test = np.clip(a, 0, 1)

        middlepost =   str(
            path.split("/")[-1].split(".png")[0])+str(current_epoch) + "_Sub_"
        utils.save_image(middlepost, a3, self.cfg.output_dir)

        a3 = utils.torch_to_np(mask)

        middlepost =  str(
            path.split("/")[-1].split(".png")[0])+str(current_epoch) + "_Mask_"
        utils.save_image(middlepost, a3, self.cfg.output_dir)

        m = torch.where(m_< ones_,ones,zeros)
        a5 = utils.torch_to_np(m)

        middlepost =  str(
            path.split("/")[-1].split(".png")[0]) + str(current_epoch) + "_TransM_"
        utils.save_image(middlepost, a5, self.cfg.output_dir)


    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):

        batch_size = img_emb_l.shape[0]
        similarities = []
        att_maps = []
        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            word = (
                text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = VSWL.VSWL.loss.VSWL_loss.attention_fn1(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]
            att_maps.append(
                attn
            )
            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = VSWL.VSWL.loss.VSWL_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities, att_maps

    def get_attn_maps(self, img_emb_l, text_emb_l, sents):
        _, _, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        return attn_maps

    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=self.cfg.data.text.word_num,
            nvis=self.cfg.train.nvis,
            rand_vis=False,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.cfg.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            print(fullpath)
            im.save(fullpath)

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):

        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def process_img(self, paths, device):

        transform = builder.build_transformation(self.cfg, split="test")

        if type(paths) == str:
            paths = [paths]

        all_imgs = []
        for p in paths:

            # x = cv2.imread(str(p), 0)
            path = []
            lcc = "path"
            lmlo = "path"
            rcc = "path"
            rmlo = "path"
            path.append(lcc)
            path.append(lmlo)
            path.append(rcc)
            path.append(rmlo)

            mask = []
            for i in range(len(path)):

                mask_tmp = cv2.imread(path[i], 0)
                mask_tmp = self._resize_img(mask_tmp, self.cfg.data.image.imsize)
                mask_tmp = Image.fromarray(mask_tmp).convert("RGB")

                mask_tmp = transform(mask_tmp)
                mask.append(mask_tmp)

            mask = torch.stack(mask)
            all_imgs.append(torch.tensor(mask))

        all_imgs = torch.stack(all_imgs).to(device)
        # print(all_imgs.shape)
        all_imgs = all_imgs.view(all_imgs.size(0) * all_imgs.size(1), all_imgs.size(2), all_imgs.size(3), all_imgs.size(4))
        print(all_imgs.shape)
        return all_imgs

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
    def plot_post1(self, m_, mask, imgs, ato, current_epoch, path):
            shape = [self.batch_size*4, 1, m_.size(2), m_.size(3)]
            t = np.zeros(shape).astype(np.float32)
            zeros = torch.zeros(1,dtype=torch.float16).cuda()
            ones = torch.ones(1,dtype=torch.float16).cuda()
            ones_ = ones * 0.58
            # ones_ = ones * 0.65
            # label_mask2 = torch.ones(label_mask2.shape, dtype=torch.long).cuda()
            ones_ = ones_.type(dtype=torch.float16)
            # trans_m = torch.where(m_ > ones_, zeros, m_)
            # trans_m = 1-m_
            for i in range(self.batch_size*4):
                t[i, :, :, :] = utils.t_matting(imgs.cpu().numpy()[i], m_.detach().cpu().numpy()[i])
                # t[i, :, :, :] = t_matting(images_torch.cpu().numpy()[i], mask_[i].detach().cpu().numpy())
                #
            t = torch.from_numpy(t).cuda()
            # post = (imgs - (t * ato)) / (1 - t)
            post = (imgs - ((1 - t) * ato)) / t
            mean = imgs - post
            mean = (mean + 1) / 2
            post = (post + 1) / 2
            post = torch.clip(post, 0, 1)
            imgs = (imgs + 1) / 2
            for i in range(imgs.size(0)):
                a = utils.torch_to_np(torch.split(imgs, 1, dim=0)[i])
                print("--------------:", np.max(a))
                # rr=post.detach().cpu().numpy()
                # test = np.clip(a, 0, 1)
                middlepost = str(
                    path[i].split("/")[-1].split(".png")[0]) +str(current_epoch) + "_input_"
                utils.save_image(middlepost, a, self.cfg.output_dir)

                a = utils.torch_to_np(torch.split(post, 1, dim=0)[i])
                print("--------------:", np.max(a))
                # rr=post.detach().cpu().numpy()
                # test = np.clip(a, 0, 1)

                middlepost = str(
                    path[i].split("/")[-1].split(".png")[0]) + str(current_epoch) + "_Post_"
                utils.save_image(middlepost, a, self.cfg.output_dir)

                a2 = utils.torch_to_np(torch.split(m_, 1, dim=0)[i])
                # rr=post.detach().cpu().numpy()
                # test = np.clip(a, 0, 1)

                middlepost =  str(
                    path[i].split("/")[-1].split(".png")[0]) +str(current_epoch) + "_M_"
                utils.save_image(middlepost, a2, self.cfg.output_dir)

                a2 = utils.torch_to_np(torch.split((1 - m_), 1, dim=0)[i])
                # rr=post.detach().cpu().numpy()
                # test = np.clip(a, 0, 1)

                middlepost =str(
                    path[i].split("/")[-1].split(".png")[0])+ str(current_epoch) + "_1-M_"
                utils.save_image(middlepost, a2, self.cfg.output_dir)

                a3 = utils.torch_to_np(torch.split(mean, 1, dim=0)[i])
                # rr=post.detach().cpu().numpy()
                # test = np.clip(a, 0, 1)

                middlepost =   str(
                    path[i].split("/")[-1].split(".png")[0])+str(current_epoch) + "_Sub_"
                utils.save_image(middlepost, a3, self.cfg.output_dir)

                a3 = utils.torch_to_np(torch.split(mask, 1, dim=0)[i])

                middlepost =  str(
                    path[i].split("/")[-1].split(".png")[0])+str(current_epoch) + "_Mask_"
                utils.save_image(middlepost, a3, self.cfg.output_dir)

                m = torch.where(m_< ones_,ones,zeros)
                a5 = utils.torch_to_np(torch.split(m, 1, dim=0)[i])

                middlepost =  str(
                    path[i].split("/")[-1].split(".png")[0]) + str(current_epoch) + "_TransM_"
                utils.save_image(middlepost, a5, self.cfg.output_dir)