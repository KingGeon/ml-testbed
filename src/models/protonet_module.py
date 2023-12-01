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
from torch.distributions.dirichlet import Dirichlet
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
        self.train_fake_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.train_mixed_up_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.train_mixedup_discriminator_real_acc= Accuracy(task = "binary")
        self.train_mixedup_discriminator_mixedup_acc= Accuracy(task = "binary")

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
        embeddings = model.forward(data)
        embeddings = model.fault_classfier(embeddings)
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
    
    def fast_fake_adapt(self, model, batch, ways, shot, mode, metric=None, device=None):
        if metric is None:
            metric = self.pairwise_distances_logits
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)

        # Sort data samples by labels
        # TODO: Can this be replaced by ConsecutiveLabels ?
        alpha = 0.0001 + torch.rand(1) * 0.0002
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)
        # Compute support and query embeddings
        embeddings = model.forward(data)
        embeddings = model.fault_classfier(embeddings)
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + shot)
        for offset in range(shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        fake_data = self.net.fake_generator(data[support_indices],alpha)
        fake_embeddings = model.forward(fake_data)
        fake_embeddings = model.fault_classfier(fake_embeddings)
        fake_datasupport = fake_embeddings.reshape(ways, shot, -1).mean(dim=1)
        query = embeddings[query_indices]
        labels = labels[query_indices].long()
        logits = self.pairwise_distances_logits(query, fake_datasupport)
 

        return logits, labels
    
    def make_mixedup(self, data, labels, alpha=0.5):
        mixed_sample_list = []
        label_list = []
        for label in labels.unique():
            # Find indices of samples with the same label
            same_label_indices = (labels == label).nonzero(as_tuple=True)[0]
            same_label_data = data[same_label_indices]
            numberofdata_in_same_label = same_label_data.size(0)

            if numberofdata_in_same_label > 1:
                # Sample weights from a Dirichlet distribution
                weights = Dirichlet(torch.tensor([alpha] * numberofdata_in_same_label)).sample()

                # Initialize mixed sample
                mixed_sample = torch.zeros_like(same_label_data[0])

                # Perform the mixup
                for i in range(numberofdata_in_same_label):
                    mixed_sample += weights[i] * same_label_data[i]

                mixed_sample_list.append(mixed_sample.unsqueeze(0))
                label_list.append(label)
        
        if mixed_sample_list:
            a = torch.cat(mixed_sample_list, dim=0)
            b = torch.tensor(label_list)
            return a, b
 
    
    def fast_adapt_mixedup_data(self, model, batch, ways, shot, mode, metric=None, device=None):
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
        embeddings = model.forward(data)
        embeddings = model.fault_classfier(embeddings)
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + shot)
        for offset in range(shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        
        support_data = data[support_indices]
        support_label = labels[support_indices]
        mixed_data, _ = self.make_mixedup(support_data,support_label)
        mixed_imbedding = model.forward(mixed_data)
        mixed_imbedding = model.fault_classfier(mixed_imbedding)
        proto = mixed_imbedding.reshape(ways, 1, -1).mean(dim=1)
        query = embeddings[query_indices]
        labels = labels[query_indices].long()
        logits = self.pairwise_distances_logits(query, proto)


        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self.fast_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        classification_loss = F.cross_entropy(logits, labels)

        #mixedup_logits, mixedup_labels  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        #mixedup_classification_loss = F.cross_entropy(mixedup_logits, mixedup_labels)
        #fake_logit, fake_labels = self.fast_fake_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        #fake_classification_loss = F.cross_entropy(fake_logit, fake_labels)
        '''
        anchor, positive, negative = self.net.prepare_triplet(batch)
        # Feature 추출
        anchor_feature = self.net.forward(anchor)
        positive_feature = self.net.forward(positive)    
        negative_feature = self.net.forward(negative)
        # Loss 계산
        triplet_loss = self.net.triplet_loss(anchor_feature, positive_feature, negative_feature)
        '''
        # 전체 손실
        total_loss =  classification_loss #+ fake_classification_loss + mixedup_classification_loss 

        self.train_loss(total_loss)
        self.train_acc(logits, labels)
        #self.train_mixed_up_acc(mixedup_logits, mixedup_labels)
        #self.train_fake_acc(fake_logit, fake_labels)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train_fake/acc', self.train_fake_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train_mixedup/acc', self.train_mixed_up_acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        logits, labels = self.fast_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        classification_loss = F.cross_entropy(logits, labels)


        self.val_loss(classification_loss)
        self.val_acc(logits, labels)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True), 
        
    def test_step(self, batch, batch_idx):
        logits, labels = self.fast_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        classification_loss = F.cross_entropy(logits, labels)
        self.test_loss(classification_loss)
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