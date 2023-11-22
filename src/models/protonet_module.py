from typing import Any
import torch.nn.functional as F
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.meta_utils import split_batch
import lightning as L
import torch
import torch.optim as optim
import numpy as np

class ProtoNetModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 N_WAY: int = 4,
                 K_SHOT: int = 4):
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        self.train_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.val_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.test_acc = Accuracy(task="multiclass", num_classes= N_WAY)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()

    def pairwise_distances_logits(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        logits = -((a.unsqueeze(1).expand(n, m, -1) -
                    b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
        return logits


    def accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)
    
    def fast_adapt(self, model, batch, ways, shot, mode, metric=None, device=None):
        if metric is None:
            metric = self.pairwise_distances_logits
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)

        # Sort data samples by labels
        # TODO: Can this be replaced by ConsecutiveLabels ?
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)
        # Compute support and query embeddings
        embeddings = model(data)
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + shot)
        for offset in range(shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices]
        support = support.reshape(ways, shot, -1).mean(dim=1)
        query = embeddings[query_indices]
        labels = labels[query_indices].long()
        logits = self.pairwise_distances_logits(query, support)
 

        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self.fast_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        loss = F.cross_entropy(logits, labels)
        self.train_loss(loss)
        self.train_acc(logits, labels)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        logits, labels = self.fast_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "val")
        loss = F.cross_entropy(logits, labels)
        self.val_loss(loss)
        self.val_acc(logits, labels)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True), 
        
    def test_step(self, batch, batch_idx):
        logits, labels = self.fast_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "test")
        loss = F.cross_entropy(logits, labels)
        self.test_loss(loss)
        self.test_acc(logits, labels)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

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
    _ = ProtoNetModule(None, None, None)
