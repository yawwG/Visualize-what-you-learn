import torch
import json
from numpy import *
from .. import builder
from pytorch_lightning.core import LightningModule
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
thresh_TIOU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
thresh_TIOR = [0.1, 0.25, 0.5, 0.75, 0.9]
class PretrainModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.VSWL = builder.build_VSWL_model(cfg)

        self.lr = cfg.lightning.trainer.lr
        # self.count = 0
        self.dm = None
        self.best_epoch = None
        self.IOU = []
        self.Dice = []
        self.counts = np.zeros(1)
        self.TIOU = np.zeros((1, len(thresh_TIOU)))
        self.best_Dice = 0.05
        self.similarities = []
        self.local_similarities = []
        self.global_similarities = []
        self.file_path =[]
        self.features_g = []
        self.features_l = []
        self.tfeatures_g = []
        self.tfeatures_l = []
        self.labels = []

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.VSWL)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # loss, attn_maps, sents, m, y1 = self.shared_step(batch, "train")
        img_emb_l, img_emb_g, text_emb_l, text_emb_g, loss, attn_maps1, sents, m, y1 = self.shared_step1(batch, "train")
        self.features_g.append(img_emb_g)  # [b,768]
        self.features_l.append(img_emb_l)  # [b,768,32,32]
        self.tfeatures_g.append(text_emb_g)  # [b,768]
        self.tfeatures_l.append(text_emb_l)  # [b,768,97]
        self.labels.append(batch['label'])
        # if batch_idx == len(batch)-1:
        if batch_idx == 89:
            # tsne
            self.features_g = torch.stack(self.features_g)
            self.features_l = torch.stack(self.features_l)  # [9,8,768,32,32]
            self.tfeatures_g = torch.stack(self.tfeatures_g)
            self.tfeatures_l = torch.stack(self.tfeatures_l)
            gfeature_bank = torch.cat((self.features_g, self.tfeatures_g), 0)
            gfeature_bank = gfeature_bank.view(gfeature_bank.size(0) * gfeature_bank.size(1), gfeature_bank.size(2))
            self.tfeatures_l = self.tfeatures_l.view(self.tfeatures_l.size(0) * self.tfeatures_l.size(1),
                                                     self.tfeatures_l.size(2) * self.tfeatures_l.size(3))  # [9*8,768*97]
            self.features_l = self.features_l.view(self.features_l.size(0) * self.features_l.size(1),
                                                   self.features_l.size(2) * self.features_l.size(
                                                       3) * self.features_l.size(4))
            # print(self.features_l.shape)
            self.features_l = self.features_l[:, 0:self.tfeatures_l.size(1)]
            # print(self.features_l.shape)
            lfeature_bank = torch.cat((self.features_l, self.tfeatures_l), 0)

            self.labels = torch.stack(self.labels)
            self.labels = self.labels.view(self.labels.size(0) * self.labels.size(1))
            self.i_patient_label = self.labels
            self.t_patient_label = self.labels + 2
            feature_labels = torch.cat((self.i_patient_label, self.t_patient_label), 0)

            print(gfeature_bank.shape)
            print(feature_labels.shape)
            self.tsne_method(gfeature_bank, feature_labels, self.current_epoch, "g")
            self.tsne_method(lfeature_bank, feature_labels, self.current_epoch, "l")
            #
            self.features_g = []
            self.features_l = []
            self.tfeatures_g = []
            self.tfeatures_l = []
        return loss

    def validation_step(self, batch, batch_idx):
        # loss, attn_maps, sents, m, y1 = self.shared_step(batch, "val")
        img_emb_l, img_emb_g, text_emb_l, text_emb_g, loss, attn_maps1, sents, m, y1 = self.shared_step1(batch, "val")
        # TIOU, mean_IOU, mean_Dice, counts = self.calculateIou(batch, m, y1)
        self.calculateIou(batch, m, y1)
        self.labels.append(batch['label'])
        if batch_idx == len(batch)-1:
            results_csv = os.path.join(self.cfg.output_dir, "seg.csv")
            # fr = open(results_csv, 'a')
            for j in range(len(thresh_TIOU)):
                if self.counts[0] == 0:
                    self.TIOU[0][j] = 0.
                else:
                    self.TIOU[0][j] = float(self.TIOU[0][j]) / float(self.counts[0])
            mean_IOU = sum(self.IOU) / sum(self.counts)
            mean_Dice = sum(self.Dice) / sum(self.counts)
                # return TIOU, mean_IOU, mean_Dice, counts
            results = {"epoch": self.current_epoch, "TIOU": self.TIOU.tolist(), "mean_IOU": mean_IOU, "mean_Dice":mean_Dice, "Counts": self.counts[0], "bestDIce": self.best_Dice, "bestEpoch": self.best_epoch}
            print(results)
            with open(results_csv, "a") as fp:
                json.dump(results, fp)
                json.dump("/n", fp)
            if(self.best_Dice<mean_Dice):
                self.best_Dice = mean_Dice
                self.best_epoch = self.current_epoch

            self.IOU = []
            self.Dice = []
            self.counts = np.zeros(1)
            self.TIOU = np.zeros((1, len(thresh_TIOU)))

        return loss

    def test_step(self, batch, batch_idx):

        ckpt = torch.load("path")
        ckpt_dict = ckpt["state_dict"]
        fixed_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_key = k.split("VSWL2.")[-1]
            fixed_ckpt_dict[new_key] = v
        ckpt_dict = fixed_ckpt_dict
        VSWL_model = self.VSWL.load_state_dict(ckpt_dict)
        img_emb_l, img_emb_g, text_emb_l, text_emb_g, loss, attn_maps1, sents, m, y1 = self.shared_step1(batch, "val")

        #tsne
        self.features_g.append(img_emb_g)  # [b,768]
        self.features_l.append(img_emb_l)  # [b,768,32,32]
        self.tfeatures_g.append(text_emb_g)  # [b,768]
        self.tfeatures_l.append(text_emb_l)  # [b,768,97]
        self.labels.append(batch['label'])

        dice, iou = self.get_dice(m, batch["mask"])
        self.IOU.append(iou)
        self.Dice.append(dice)

        if batch_idx == 89:
            # results_csv = os.path.join(self.cfg.output_dir, "seg.csv")
            for j in range(len(thresh_TIOU)):
                if self.counts[0] == 0:
                    self.TIOU[0][j] = 0.
                else:
                    self.TIOU[0][j] = float(self.TIOU[0][j]) / float(self.counts[0])
            #         tsne
            self.features_g = torch.stack(self.features_g)
            self.features_l = torch.stack(self.features_l)  # [9,8,768,32,32]
            self.tfeatures_g = torch.stack(self.tfeatures_g)
            self.tfeatures_l = torch.stack(self.tfeatures_l)
            gfeature_bank = torch.cat((self.features_g, self.tfeatures_g), 0)
            gfeature_bank = gfeature_bank.view(gfeature_bank.size(0) * gfeature_bank.size(1), gfeature_bank.size(2))
            self.tfeatures_l = self.tfeatures_l.view(self.tfeatures_l.size(0) * self.tfeatures_l.size(1),
                                                     self.tfeatures_l.size(2) * self.tfeatures_l.size(
                                                         3))  # [9*8,768*97]
            self.features_l = self.features_l.view(self.features_l.size(0) * self.features_l.size(1),
                                                   self.features_l.size(2) * self.features_l.size(
                                                       3) * self.features_l.size(
                                                       4))
            self.features_l = self.features_l[:, 0:self.tfeatures_l.size(1)]
            lfeature_bank = torch.cat((self.features_l, self.tfeatures_l), 0)

            self.labels = torch.stack(self.labels)
            self.labels = self.labels.view(self.labels.size(0) * self.labels.size(1))
            self.i_patient_label = self.labels
            self.t_patient_label = self.labels + 2
            feature_labels = torch.cat((self.i_patient_label, self.t_patient_label), 0)

            print(gfeature_bank.shape)
            print(feature_labels.shape)
            self.tsne_method(gfeature_bank, feature_labels, self.current_epoch, "g")
            self.tsne_method(lfeature_bank, feature_labels, self.current_epoch, "l")
            #
            self.features_g = []
            self.features_l = []
            self.tfeatures_g = []
            self.tfeatures_l = []

        mean_IOU = mean(self.IOU)

        mean_Dice = mean(self.Dice)

        results = {"bach": batch_idx, "TIOU": self.TIOU.tolist(), "mean_IOU": mean_IOU,
                   "mean_Dice": mean_Dice}
        print(results)

        return loss

    def get_dice(self, probability, truth, threshold=0.5):
        # threshold = 0.2
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            print(probability.shape)
            print(truth.shape)
            assert probability.shape == truth.shape

            p = (probability < threshold).float()
            t = (truth > threshold).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

            dice_neg = dice_neg[neg_index]
            # dice_neg = dice_pos[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])
            # dice = dice_pos
            iOU = dice / (2 - dice)
            for j in range(len(thresh_TIOU)):
                if torch.mean(iOU) >= thresh_TIOU[j]:
                    self.TIOU[0][j] += 1
            self.counts += 1

        return torch.mean(dice).detach().item(), torch.mean(iOU).detach().item()

    def tsne_method(self, feature_bank, feature_labels, epoch, flag):

        memory_feature_bank = feature_bank
        memory_feature_labels = feature_labels
        # memory_feature_bank = memory_feature_bank.cpu()
        # memory_feature_labels = memory_feature_labels.cpu()
        i_memory_feature_bank = []
        i_memory_feature_labels = []
        for orderi in range(0, len(memory_feature_bank)):
            i_memory_feature_bank.append(memory_feature_bank[orderi].tolist())
            i_memory_feature_labels.append(memory_feature_labels[orderi].tolist())

        memory_feature_bank = np.array(i_memory_feature_bank)
        print(memory_feature_bank.shape[0], memory_feature_bank.shape[1])
        # memory_feature_bank = memory_feature_bank.view
        memory_feature_bank = memory_feature_bank.reshape(-1, memory_feature_bank.shape[1])
        print(memory_feature_bank.shape)
        memory_feature_labels = np.array(i_memory_feature_labels)
        memory_feature_labels = memory_feature_labels.reshape(-1, 1)

        # t-sne
        tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
        result_2D = tsne_2D.fit_transform(memory_feature_bank)
        classesoflabes = np.unique(memory_feature_labels)

        color = ['red', 'orange', 'blue', 'purple', 'yellow']
        marker = ['x', 'o', 'x', 'o', 's']

        plt.figure()
        for i in range(0, len(classesoflabes)):
            result_sub = result_2D[np.where(memory_feature_labels == classesoflabes[i])[0], :]
            plt.scatter(result_sub[:, 0], result_sub[:, 1], color=color[i], marker=marker[i], alpha=0.5,
                        label=str(classesoflabes[i]))

        # plt.title('t-SNE' + '-epoch' + str(epoch))
        plt.legend(loc='upper right')
        if flag == "g":
            plt.savefig(self.cfg.output_dir + '/epoch' + str(epoch) + '-t-SNE_global_acr.png')
            print(self.cfg.output_dir + '/epoch' + str(epoch) + '-t-SNE_global_ar.png')
        else:
            plt.savefig(self.cfg.output_dir + '/epoch' + str(epoch) + '-t-SNE_local_acr.png')

        # plt.savefig(args.results_dir + '/epoch' + str(epoch) + '-t-SNE.png', dpi=300)
        plt.close()
    def calculateIou(self,batch, m, y1):

        m = m <= 0.58
        for jdx in range(m.size(0)):
            # tmp = batch["masks"][jdx, :, :, :]
            mask_path = "path"
            if (os.path.exists(mask_path)):
                    batch_iou = self.single_IOU(m[jdx, 0 , :, :],  batch["mask"][jdx, 0, :, :])
                    # batch_ior = single_IOR(cam_classes_refined[jdx, idx, :, :], masks[jdx, idx, :, :])
                    batch_dice = self.single_Dice(m[jdx, 0 , :, :],  batch["mask"][jdx, 0, :, :])
                    self.IOU.append(batch_iou)
                    # IOR_CAM.append(batch_ior)
                    self.Dice.append(batch_dice)
                    for j in range(len(thresh_TIOU)):
                        if batch_iou >= thresh_TIOU[j]:
                            self.TIOU[0][j] += 1
                    self.counts[0] +=1

    def single_IOU(self, pred, target):

        pred_class = pred.data.cpu().contiguous().view(-1)
        target_class = target.data.cpu().contiguous().view(-1)
        pred_inds = pred_class > 0
        target_inds = target_class > 0

        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        iou = float(intersection) / float(max(union, 1))
        return iou

    def single_Dice(self,pred, target):

        pred_class = pred.data.cpu().contiguous().view(-1)
        target_class = target.data.cpu().contiguous().view(-1)
        pred_inds = pred_class > 0
        target_inds = target_class > 0

        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union_and_intersection = pred_inds.long().sum().item() + target_inds.long().sum().item()
        # print(float(intersection))
        # print(float(union))
        Dice = 2 * float(intersection) / float(max(union_and_intersection, 1))
        return Dice

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, m, y1, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2 = self.VSWL(batch)
        loss, attn_maps = self.VSWL.calc_loss(
            img_emb_l, img_emb_g, m, y1, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, batch, self.current_epoch
        )

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return loss, attn_maps, sents, m, y1
    def shared_step1(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, m, y1, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2 = self.VSWL(batch)
        loss, attn_maps = self.VSWL.calc_loss(
            img_emb_l, img_emb_g, m, y1, text_emb_l, text_emb_g, sents, text_mlp1, text_mlp2, batch, self.current_epoch
        )

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, loss, attn_maps, sents, m, y1

