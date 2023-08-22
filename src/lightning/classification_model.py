import numpy as np
import torch
import json
import os
import csv
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, roc_auc_score
from .. import builder
from .. import VSWL
from pytorch_lightning.core import LightningModule


class ClassificationModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        if self.cfg.model.vision.model_name in VSWL.available_models():
            self.model = VSWL.load_img_classification_model(
                self.cfg.model.vision.model_name,
                num_cls=self.cfg.model.vision.num_targets,
                freeze_encoder=self.cfg.model.vision.freeze_cnn,
            )
        else:
            self.model = builder.build_img_model(cfg)
        try:
            self.cfg.test_path = cfg.test_path
        except:
            print('')
        else:
            ckpt = torch.load(self.cfg.test_path)
            ckpt_dict = ckpt["state_dict"]
            net_dict = self.model.state_dict()
            fixed_ckpt_dict = {}
            i = 0
            for k, v in ckpt_dict.items():
                new_key = list(net_dict.keys())[i]
                fixed_ckpt_dict[new_key] = v
                i += 1
            net_dict.update(fixed_ckpt_dict)  # update dict
            self.model.load_state_dict(net_dict)  # model.load_state_dict()
            print('load test ckpt!')
            
        self.loss = builder.build_loss(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.best_auroc = 0.5
        self.best_epoch = 0

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split):
        """Similar to traning step"""

        x, y = batch
        # y = y.type(dtype=torch.long)
        logit = self.model(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(logit, y.float())
        # print(classification_report(y.cpu().numpy(), logit.cpu().numpy(), digits=3))
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "logit": logit, "y": y}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        logit = torch.cat([x["logit"] for x in step_outputs])
        y = torch.cat([x["y"] for x in step_outputs])
        prob = torch.sigmoid(logit)

        y = y.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()

        auroc_list, auprc_list = [], []
        for i in range(y.shape[1]):
            y_cls = y[:, i]
            prob_cls = prob[:, i]

            if np.isnan(prob_cls).any():
                auprc_list.append(0)
                auroc_list.append(0)
            else:
                auprc_list.append(average_precision_score(y_cls, prob_cls))
                try:
                    auroc_list.append(roc_auc_score(y_cls, prob_cls))
                except:
                    print("auc not exist!")

        auprc = np.mean(auprc_list)
        auroc = np.mean(auroc_list)
        prob_ = np.where(prob > 0.5, 1, 0)
        print(classification_report(y[:, 0], prob_[:, 0], digits=3))
        self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)
        probsize = prob.shape[0]
        # print("prob_size:", prob.shape,probsize )
        if split != 'train':
            if (self.best_auroc < auroc) and probsize > 16:
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                with open(self.cfg.output_dir + 'pro.csv', 'w+', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')
                    mywriter.writerows(prob)
                    file.close()
                with open(self.cfg.output_dir + 'label.csv', 'w+', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')
                    mywriter.writerows(y)
                    file.close()
            results = {"epoch": self.current_epoch, "auroc": auroc, "bestauc": self.best_auroc,
                       "bestEpoch": self.best_epoch}
            results_csv = os.path.join(self.cfg.output_dir, "results.csv")
            with open(results_csv, "a") as fp:
                json.dump(results, fp)
                json.dump("/n", fp)