import torch
import pytorch_lightning as pl

import segmentation_models_pytorch as smp

import torchmetrics 
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryAccuracy


class SegmentationModule(pl.LightningModule):
    def __init__(self, model, loss, learning_rate = 3e-4, batch_size = 32):
        super().__init__()
        self.model = model

        assert loss in ["dice", "iou", "bce"]
        if loss == "dice":
            self.loss = smp.losses.DiceLoss(mode = "binary")
        elif loss == "iou":
            self.loss = smp.losses.JaccardLoss(mode = "binary")
        elif loss == "bce":
            self.loss = smp.losses.SoftBCEWithLogitsLoss()

        self.save_hyperparameters(ignore = ["model", "preprocess"])

        self.val_metrics = MetricCollection([
            BinaryF1Score(),
            BinaryJaccardIndex(),
            BinaryAccuracy()
        ], prefix = "Validation ")

        self.test_metrics = MetricCollection([
            BinaryF1Score(),
            BinaryJaccardIndex(),
            BinaryAccuracy()
        ], prefix = "Test ")

        self.pred_metrics = MetricCollection([
            BinaryF1Score(),
            BinaryJaccardIndex(),
            BinaryAccuracy()
        ], prefix = "Pred ")


    def forward(self, batch):
        mask = self.model(batch) 
        return mask
        
    def _shared_forward_pass(self, batch):
        scene, mask = batch
        pred_mask = self.model(scene)
        return pred_mask, mask

    def training_step(self, batch, batch_idx):
        pred_mask, mask = self._shared_forward_pass(batch)
        loss = self.loss(pred_mask, mask) 
        self.log("train/loss", loss.item(), on_epoch = True, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred_mask, mask = self._shared_forward_pass(batch)
        loss = self.loss(pred_mask, mask)
        self.log("val/loss", loss.item(), on_epoch=True)

        self.val_metrics.update(pred_mask, mask)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pred_mask, mask = self._shared_forward_pass(batch)
        loss = self.loss(pred_mask, mask)
        self.log("test/loss", loss.item(), on_epoch=True)

        self.test_metrics.update(pred_mask, mask)
        self.log_dict(self.test_metrics, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        pred_mask, mask = self._shared_forward_pass(batch)
        return pred_mask, mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hparams.learning_rate)
        return optimizer