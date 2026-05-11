from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import numpy as np
import sys

def calculate_miou(preds, targets, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return sum(ious) / num_classes

class SSCLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        dataset: str,
        class_num: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.data_set = dataset

        self.net = net

        if "V2XSeqSPD" in self.data_set:
            print('torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)\n' * 5)
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.preds_list = []
            self.targets_list = []

        # loss function
        self.criterion = loss

        self.sync_dist = "V2XSeqSPD" in self.data_set

        # metric objects for calculating and averaging acc,miou across batches
        self.train_mAcc = MulticlassAccuracy(class_num)
        self.train_mIoU = MulticlassJaccardIndex(class_num)
        self.val_mAcc = MulticlassAccuracy(class_num)
        self.val_mIoU = MulticlassJaccardIndex(class_num)
        self.test_mAcc = MulticlassAccuracy(class_num)
        self.test_mIoU = MulticlassJaccardIndex(class_num)
        self.per_class_test_mAcc = MulticlassAccuracy(class_num, average=None)
        self.per_class_test_mIoU = MulticlassJaccardIndex(class_num, average=None)

        # for averaging loss across batches
        self.train_cd = MeanMetric()
        self.train_seg = MeanMetric()
        self.val_cd = MeanMetric()
        self.val_seg = MeanMetric()
        self.test_cd = MeanMetric()
        self.test_seg = MeanMetric()
        self.test_f_score = MeanMetric()

        self.val_cd_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.net.train()  # 切换到评估模式
        self.val_cd_best.reset()

    def on_test_start(self):
        torch.cuda.empty_cache()  # 手动清理缓存
        print('on_test_start')

    def on_validation_start(self):
        torch.cuda.empty_cache()  # 手动清理缓存
        print('on_validation_start')

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y, self.current_epoch)
        sum_loss, last_cd, last_seg, f_score = (
            loss["sum_loss"],
            loss["last_cd"],
            loss["last_seg"],
            loss["f_score"],
        )
        preds, gt_seg = loss["pred_seg"], loss["gt_seg"]

        return (sum_loss, last_cd, last_seg, f_score), preds, gt_seg

    def training_step(self, batch: Any, batch_idx: int):
        # if batch_idx % 100 != 0:
        #     return
        loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg, last_f_score = loss[0], loss[1], loss[2], loss[3]

        if "V2XSeqSPD" in self.data_set:
            b, n, cls = preds.shape
            preds = preds.contiguous().reshape(b * n, cls)
            targets = targets.contiguous().reshape(-1)
            valid_indices = targets != 18  # 根据条件得到布尔掩码
            preds = preds[valid_indices, :18]  # 从preds中选择有效的位置
            targets = targets[valid_indices]  # 从targets中选择有效的位置
            max_values, preds = torch.max(preds, dim=1)
        else:
            preds = preds[:, :, 1:] if "nyucad" in self.data_set else preds
            preds = preds.transpose(1, 2)
            targets = targets - 1 if "nyucad" in self.data_set else targets
        # update and log metrics
        self.train_cd(last_cd)
        self.train_seg(last_seg)
        # targets = targets if self.data_set == "ssc_pc" or self.data_set == "3D-FRONT_pc" or "V2XSeqSPD" in self.data_set else targets - 1
        _ = self.train_mAcc(preds, targets)
        _ = self.train_mIoU(preds, targets)
        self.log("train/cd", self.train_cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log(
            "train/seg", self.train_seg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )
        self.log(
            "train/mAcc", self.train_mAcc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )
        self.log(
            "train/mIoU", self.train_mIoU, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():  # 禁用梯度计算
            loss, preds, targets = self.model_step(batch)
            preds = preds.detach()  # 分离预测结果
            targets = targets.detach()  # 分离目标数据
            del batch
            # torch.cuda.empty_cache()
            # print('shape' * 100, preds.shape, targets.shape)
            loss, last_cd, last_seg, last_f_score = loss[0], loss[1], loss[2], loss[3]

            if "V2XSeqSPD" in self.data_set:
                b, n, cls = preds.shape
                preds = preds.contiguous().reshape(b * n, cls)
                targets = targets.contiguous().reshape(-1)
                valid_indices = targets != 18  # 根据条件得到布尔掩码
                preds = preds[valid_indices, :18]  # 从preds中选择有效的位置
                targets = targets[valid_indices]  # 从targets中选择有效的位置
                max_values, preds = torch.max(preds, dim=1)
                del valid_indices
                del max_values
                # self.preds_list.append(preds)
                # self.targets_list.append(targets)
                self.val_cd(last_cd)
                self.val_seg(last_seg)
                _ = self.val_mAcc(preds, targets)
                _ = self.val_mIoU(preds, targets)
                self.log("val/cd", self.val_cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
                self.log("val/seg", self.val_seg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
                self.log("val/mAcc", self.val_mAcc, on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=self.sync_dist)
                self.log("val/mIoU", self.val_mIoU, on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=self.sync_dist)
            else:
                preds = preds[:, :, 1:] if "nyucad" in self.data_set else preds
                preds = preds.transpose(1, 2)
                # update and log metrics
                self.val_cd(last_cd)
                targets = targets - 1 if "nyucad" in self.data_set else targets
                self.val_seg(last_seg)
                _ = self.val_mAcc(preds, targets)
                _ = self.val_mIoU(preds, targets)
                self.log("val/cd", self.val_cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
                self.log("val/seg", self.val_seg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
                self.log("val/mAcc", self.val_mAcc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
                self.log("val/mIoU", self.val_mIoU, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)

            # if batch_idx % 100 == 0:
            #     torch.cuda.empty_cache()  # 手动清理缓存
            # return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        if "V2XSeqSPD" in self.data_set:
            # preds = torch.cat(self.preds_list, dim=0)  # 拼接所有批次的 preds
            # targets = torch.cat(self.targets_list, dim=0)  # 拼接所有批次的 targets
            # self.preds_list.clear()
            # self.targets_list.clear()
            # print('preds   ' * 10, preds[:300].cpu().tolist())
            # print('targets ' * 10, targets[:300].cpu().tolist())
            # miou = calculate_miou(preds, targets, 18)
            # print('miou   ' * 10, miou)
            val_cd = self.val_cd.compute()
            self.val_cd_best(val_cd)  # update best so far val acc
            val_best = self.val_cd_best.compute()
            if val_cd == val_best:
                # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
                # otherwise metric would be reset by lightning after each epoch
                self.log("val_best/cd", self.val_cd_best.compute(), prog_bar=True, sync_dist=self.sync_dist)
                self.log("val_best/seg", self.val_seg.compute(), prog_bar=True, sync_dist=self.sync_dist)
                self.log("val_best/mAcc", self.val_mAcc.compute(), prog_bar=True, sync_dist=self.sync_dist)
                self.log("val_best/mIoU", self.val_mIoU.compute(), prog_bar=True, sync_dist=self.sync_dist)
        else:
            val_cd = self.val_cd.compute()
            self.val_cd_best(val_cd)  # update best so far val acc
            val_best = self.val_cd_best.compute()
            if val_cd == val_best:
                # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
                # otherwise metric would be reset by lightning after each epoch
                self.log("val_best/cd", self.val_cd_best.compute(), prog_bar=True, sync_dist=self.sync_dist)
                self.log("val_best/seg", self.val_seg.compute(), prog_bar=True, sync_dist=self.sync_dist)
                self.log("val_best/mAcc", self.val_mAcc.compute(), prog_bar=True, sync_dist=self.sync_dist)
                self.log("val_best/mIoU", self.val_mIoU.compute(), prog_bar=True, sync_dist=self.sync_dist)

    def _model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y, self.current_epoch)
        sum_loss, last_cd, last_seg, f_score = (
            loss["sum_loss"],
            loss["last_cd"],
            loss["last_seg"],
            loss["f_score"],
        )
        preds, gt_seg = loss["pred_seg"], loss["gt_seg"]

        return (sum_loss, last_cd, last_seg, f_score), preds, gt_seg, logits

    def _test_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():  # 禁用梯度计算
            loss, preds, targets = self.model_step(batch)
            preds = preds.detach()  # 分离预测结果
            targets = targets.detach()  # 分离目标数据
            del batch
            # torch.cuda.empty_cache()
            # print('shape' * 100, preds.shape, targets.shape)
            loss, last_cd, last_seg, last_f_score = loss[0], loss[1], loss[2], loss[3]

            if "V2XSeqSPD" in self.data_set:
                b, n, cls = preds.shape
                preds = preds.contiguous().reshape(b * n, cls)
                targets = targets.contiguous().reshape(-1)
                valid_indices = targets != 18  # 根据条件得到布尔掩码
                preds = preds[valid_indices, :18]  # 从preds中选择有效的位置
                targets = targets[valid_indices]  # 从targets中选择有效的位置
                max_values, preds = torch.max(preds, dim=1)
                del valid_indices
                del max_values
                # self.preds_list.append(preds)
                # self.targets_list.append(targets)
                self.test_cd(last_cd)
                self.test_seg(last_seg)
            else:
                preds = preds[:, :, 1:] if "nyucad" in self.data_set else preds
                preds = preds.transpose(1, 2)
                # update and log metrics
                self.test_cd(last_cd)
                targets = targets - 1 if "nyucad" in self.data_set else targets
                self.test_seg(last_seg)
            self.test_f_score(last_f_score)
            _ = self.test_mAcc(preds, targets)
            _ = self.test_mIoU(preds, targets)
            self.log("test/cd", self.test_cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
            self.log("test/seg", self.test_seg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
            self.log("test/f_score", self.test_f_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
            self.log("test/mAcc", self.test_mAcc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
            self.log("test/mIoU", self.test_mIoU, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        # return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, results = self._model_step(batch)
        loss, last_cd, last_seg, last_f_score = loss[0], loss[1], loss[2], loss[3]

        if "V2XSeqSPD" in self.data_set:
            b, n, cls = preds.shape
            preds = preds.contiguous().reshape(b * n, cls)
            targets = targets.contiguous().reshape(-1)
            valid_indices = targets != 18  # 根据条件得到布尔掩码
            preds = preds[valid_indices, :18]  # 从preds中选择有效的位置
            targets = targets[valid_indices]  # 从targets中选择有效的位置
            max_values, preds = torch.max(preds, dim=1)
            del valid_indices
            del max_values
            self.test_cd(last_cd)
            self.test_seg(last_seg)
        else:
            preds = preds[:, :, 1:] if "nyucad" in self.data_set else preds
            preds = preds.transpose(1, 2)
            # update and log metrics
            self.test_cd(last_cd)
            targets = targets - 1 if "nyucad" in self.data_set else targets
            self.test_seg(last_seg)
        self.test_f_score(last_f_score)
        _ = self.test_mAcc(preds, targets)
        _ = self.test_mIoU(preds, targets)
        _ = self.per_class_test_mAcc(preds, targets)
        _ = self.per_class_test_mIoU(preds, targets)
        self.log("test/cd", self.test_cd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log("test/seg", self.test_seg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log("test/f_score", self.test_f_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log("test/mAcc", self.test_mAcc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        self.log("test/mIoU", self.test_mIoU, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        per_class_acc = self.per_class_test_mAcc.compute()
        per_class_iou = self.per_class_test_mIoU.compute()
        # 逐类别记录
        for cls in range(len(per_class_acc)):
            self.log(f"test_mAcc_class_{cls}", per_class_acc[cls], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test_mIoU_class_{cls}", per_class_iou[cls], on_step=False, on_epoch=True, prog_bar=True)
        print('per_cls macc', per_class_acc)
        print('per_cls miou', per_class_iou)

        # return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # def configure_model(self):
    #     # 将 BatchNorm 转换为 SyncBatchNorm
    #     if "V2XSeqSPD" in self.data_set:
    #         self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
