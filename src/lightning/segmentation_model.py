import numpy as np
import torch
import json
import os
import segmentation_models_pytorch as smp
import pandas as pd
import torch.nn as nn
from sklearn.metrics import classification_report
from pytorch_lightning.core import LightningModule
from .. import builder
from .. import VSWL

thresh_TIOU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
thresh_TIOR = [0.1, 0.25, 0.5, 0.75, 0.9]
class SegmentationModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()
        self.counts = np.zeros(1)
        self.TIOU = np.zeros((1, len(thresh_TIOU)))
        self.cfg = cfg
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.IOU = []
        self.Dice = []
        self.l1loss = nn.L1Loss()
        self.fc_position = nn.Linear(2048* 16 * 16, 3, bias=True)
        self.fc_location = nn.Linear(2048 * 16 * 16, 8, bias=True)
        self.best_dice = 0.5
        self.loss_criterion = nn.CrossEntropyLoss()
        self.results =  {'dice': [], 'iou':[], 'tiou':[]}
        if self.cfg.model.vision.model_name in VSWL.available_models():
            self.model = VSWL.load_img_segmentation_model(
                self.cfg.model.vision.model_name
            )
        else:
            self.model = smp.Unet("resnet50", encoder_weights=None, activation=None)
        try:
            self.cfg.test_path = cfg.test_path
        except:
            print('')
        else:
            ckpt = torch.load(self.cfg.test_path)
            ckpt_dict = ckpt["state_dict"]
            net_dict = self.model.state_dict()
            fixed_ckpt_dict = {}
            # test = pretrain['state_dict'].keys()
            i = 0
            for k, v in ckpt_dict.items():
                # if i < 318:
                new_key = list(net_dict.keys())[i]
                fixed_ckpt_dict[new_key] = v
                i += 1
            net_dict.update(fixed_ckpt_dict)
            self.model.load_state_dict(net_dict)

        self.loss = builder.build_loss(cfg)
        self.best_epoch = 0

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)

        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test", batch_idx)

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split, batch_idx):
        """Similar to traning step"""
        x, y,  maskpath, segment_label = batch
        x = torch.cat((torch.split(x,1,dim=0)[0][0],torch.split(x,1,dim=0)[1][0],torch.split(x,1,dim=0)[2][0]),dim=0)
        segment_label = torch.cat((torch.split(segment_label, 1, dim=0)[0][0], torch.split(segment_label, 1, dim=0)[1][0], torch.split(segment_label, 1, dim=0)[2][0]),
                      dim=0)
        logit1 = self.model(x)

        y = torch.cat((torch.split(y, 1, dim=0)[0][0], torch.split(y, 1, dim=0)[1][0], torch.split(y, 1, dim=0)[2][0]),
                      dim=0)
        y = y.view(y.size(0),1,1,1)
        loss = self.loss(logit1*y, segment_label*y)
        prob = torch.sigmoid(logit1)
        dice,iou = self.get_dice(prob,segment_label )
#----calculate accuracy_test
        # if batch_idx == 0:
        #     img = batch[0][0].cpu().numpy()
        #     mask = batch[1][0].cpu().numpy()
        #     mask = np.stack([mask, mask, mask])
        #
        #     layered = 0.6 * mask + 0.4 * img
        #     img = img.transpose((1, 2, 0))
        #     mask = mask.transpose((1, 2, 0))
        #     layered = layered.transpose((1, 2, 0))
        #
        #     self.logger.experiment.log(
        #         {"input_image": [wandb.Image(img, caption="input_image")]}
        #     )
        #     self.logger.experiment.log({"mask": [wandb.Image(mask, caption="mask")]})
        #     self.logger.experiment.log(
        #         {"layered": [wandb.Image(layered, caption="layered")]}
        #     )
        #     self.logger.experiment.log({"pred": [wandb.Image(prob[0], caption="pred")]})
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=False,
        )


        return_dict = {"loss": loss, "dice": dice,  "iou":iou, "tiou":self.TIOU}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        if self.cfg.lightning.trainer.distributed_backend == "dp":
            loss = (
                torch.stack([x["loss"] for x in step_outputs]).cpu().detach().tolist()
            )
        else:
            loss = [x["loss"].cpu().detach().item() for x in step_outputs]
        dice1 = [x["dice"] for x in step_outputs]
        iou1 = [x["iou"] for x in step_outputs]
        loss = np.array(loss).mean()
        tiou = [x / self.counts for x in self.TIOU]

        dice = np.array(dice1).mean()
        iou = np.array(iou1).mean()

        self.counts = np.zeros(1)
        self.TIOU = np.zeros((1, len(thresh_TIOU)))
        self.results['dice'].append(dice)
        self.results['iou'].append(iou)
        self.results['tiou'].append(tiou)
        print("dice: ", dice, "iou: ", iou, "tiou: ", tiou)
        data_frame = pd.DataFrame(data= self.results)
        data_frame.to_csv(self.cfg.output_dir+'seg.csv')
        self.log(f"{split}_dice", dice, on_epoch=True, logger=True, prog_bar=False)
        if split != "train" :
            results2 = {"epoch": self.current_epoch, "dice_each": dice1}
            results3 = {"epoch": self.current_epoch, "iou_each": iou1}
            if (self.best_dice < dice) and  self.current_epoch!=0:
                self.best_dice = dice
                self.best_epoch = self.current_epoch

                results_csv2 = os.path.join(self.cfg.output_dir, "dice_eachresults.csv")
                with open(results_csv2, "w+") as fp:
                    json.dump(results2, fp)
                    json.dump("/n", fp)
                results_csv3 = os.path.join(self.cfg.output_dir, "iou_eachresults.csv")
                with open(results_csv3, "w+") as fp:
                    json.dump(results3, fp)
                    json.dump("/n", fp)
            results = {"epoch": self.current_epoch, "dice": dice, "iou": iou, "tiou": tiou[0].tolist(),
                       "bestdice": self.best_dice,
                       "bestEpoch": self.best_epoch}
            results_csv = os.path.join(self.cfg.output_dir, "results.csv")
            with open(results_csv, "a") as fp:
                json.dump(results, fp)
                json.dump("/n", fp)


    def get_dice(self, probability, truth, threshold=0.5):

        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert probability.shape == truth.shape

            p = (probability > threshold).float()
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
            iOU = dice/(2-dice)
            for j in range(len(thresh_TIOU)):
                if torch.mean(iOU) >= thresh_TIOU[j]:
                    self.TIOU[0][j] += 1
            self.counts += 1

        return torch.mean(dice).detach().item(), torch.mean(iOU).detach().item()

    def single_IOU(self, pred, target):

        pred_class = pred.data.cpu().contiguous().view(-1)
        target_class = target.data.cpu().contiguous().view(-1)
        pred_inds = pred_class > 0
        target_inds = target_class > 0

        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        # print(float(intersection))
        # print(float(union))
        iou = float(intersection) / float(max(union, 1))
        return iou

    def single_Dice(self, pred, target):

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

    def calculateIou(self, predict, mask, maskpath):
        m = torch.where(predict >= 0.2, 1, 0)
        # for idx in range(num_classes):
        for jdx in range(m.size(0)):
            if (os.path.exists(maskpath[jdx])):

                batch_iou = self.single_IOU(m[jdx, :, :],  mask[jdx,  :, :])
                # batch_ior = single_IOR(cam_classes_refined[jdx, idx, :, :], masks[jdx, idx, :, :])
                batch_dice = self.single_Dice(m[jdx, :, :],  mask[jdx, :, :])

                self.IOU.append(batch_iou)
                # IOR_CAM.append(batch_ior)
                self.Dice.append(batch_dice)

                for j in range(len(thresh_TIOU)):
                    if batch_iou >= thresh_TIOU[j]:
                        self.TIOU[0][j] += 1
                self.counts[0] +=1
