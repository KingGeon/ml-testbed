import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class TapNetModule(nn.Module):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 N_WAY: int = 4,
                 K_SHOT: int = 4):
        super(TapNetModule, self).__init__()
        self.net = net
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        # Define the task-specific projection layer
        self.projection_layer = nn.Linear(5, self.K_SHOT, bias=False)

    
    def fast_adapt_tapnet(self, model, batch, ways, shot, mode, metric=None, device=None):
        if metric is None:
            metric = self.pairwise_distances_logits
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)
        n_items = shot * ways

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
        prototype = support.reshape(ways, shot, -1).mean(dim=1)#prototype
        query_embedding = embeddings[query_indices]
        tap_features = self.task_adaptive_projection(query_embedding, prototype)
        labels = labels[query_indices].long()

        logits = self.compute_logits(tap_features, prototype)
        loss = F.cross_entropy(logits, labels)
        acc = self.accuracy(logits, labels)

        self.log("%s/loss" % mode, loss)
        self.log("%s/acc" % mode, acc)

        return loss, acc
    

    def task_adaptive_projection(self, features, prototypes):
    # Initial projection (e.g., subtraction)
        initial_projection = features.unsqueeze(1) - prototypes.unsqueeze(0)

        # Task-specific adaptation using the linear layer
        tap_features = self.projection_layer(initial_projection)

        return tap_features

    
    def training_step(self, batch, batch_idx):
        loss, _ = self.fast_adapt_tapnet(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.fast_adapt_tapnet(self.net, batch, self.N_WAY, self.K_SHOT, mode = "val")
        return loss

    def on_validation_epoch_end(self):
         # Retrieve the current validation accuracy from the logged metrics
        current_val_acc = self.trainer.callback_metrics.get("val/acc")

        # Update the best validation accuracy metric
        self.val_acc_best.update(current_val_acc)

        # Log the best validation accuracy
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        loss, _ = self.fast_adapt_tapnet(self.net, batch, self.N_WAY, self.K_SHOT, mode = "test")
        return loss

    def on_test_epoch_end(self):
         # Retrieve the current validation accuracy from the logged metrics
        current_test_acc = self.trainer.callback_metrics.get("test/acc")

        # Update the best validation accuracy metric
        self.test_acc_best.update(current_test_acc)

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
