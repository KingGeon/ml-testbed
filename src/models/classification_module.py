from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.data.datasets.aihub_motor_vibraion_proto import *
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import seaborn as sns

class ClassificationModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler):
        super().__init__()
        self.test_conf_matrix = ConfusionMatrix(num_classes=5, task='multiclass')
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=5)
        self.val_acc = Accuracy(task="multiclass", num_classes=5)
        self.test_acc = Accuracy(task="multiclass", num_classes=5)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        
    def forward(self, x):
        return self.net(x)
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        
    def model_step(self, batch: Any):
        x, y = batch
        logits = self.net.forward(x)
        logits = self.net.fault_classfier(logits)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim = 1)
        return loss, preds, y
    
    def training_step(self, batch: Any, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True), 
        
    def test_step(self, batch, batch_index: int):
        loss, preds, targets = self.model_step(batch)
        self.test_conf_matrix(preds, targets)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self):
        conf_matrix = self.test_conf_matrix.compute()
        self.save_confusion_matrix(conf_matrix)

    def save_confusion_matrix(self, matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(matrix.cpu().numpy()), annot=True, fmt='g', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('/home/geon/dev_geon/ml-testbed/src/figures/confusion_matrix.png')
        plt.close()
    
    def configure_optimizers(self):
        optimizer=self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler(optimizer=optimizer):
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer":optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        return {"optimizer": optimizer}
    
    
if __name__ == "__main__":
    _ = ClassificationModule(None, None, None)
    
    
        