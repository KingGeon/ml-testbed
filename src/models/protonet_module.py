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

class ProtoNetModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler):
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.val_acc_best = MaxMetric()
    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes
    
    def classify_feats(self, prototypes, classes, feats, targets):
        # Classify new examples with prototypes and return classification error
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc
    
    def calculate_loss(self, batch, mode):
        # Determine training loss for a given support and query set
        x, y , targets = batch
        features = self.model(x, y)  # Encode all images of support and query set
        support_feats, query_feats, support_targets, query_targets = split_batch(features, targets)
        prototypes, classes = ProtoNetModule.calculate_prototypes(support_feats, support_targets)
        preds, labels, acc = self.classify_feats(prototypes, classes, query_feats, query_targets)
        loss = F.cross_entropy(preds, labels)

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="val")

    def on_validation_epoch_end(self):
         # Retrieve the current validation accuracy from the logged metrics
        current_val_acc = self.trainer.callback_metrics.get("val_acc")

        # Update the best validation accuracy metric
        self.val_acc_best.update(current_val_acc)

        # Log the best validation accuracy
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)


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