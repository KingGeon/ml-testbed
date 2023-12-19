from typing import Any
import librosa
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
import os
import random
import math
from src.data.datasets.aihub_motor_vibraion_proto import *
import matplotlib.pyplot as plt


class ProtoNetModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 initial_beta: float = 0.1,
                 max_beta: float = 0.99,
                 epochs_to_max: float = 20,
                 epochs_to_min: float = 10,
                 initial_alpha: float = 0.99,
                 max_alpha: float = 0.01,
                 N_WAY: int = 4,
                 K_SHOT: int = 10):
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        self.train_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.train_fake_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.train_mixed_up_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.train_reconst_mixed_up_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.train_mixedup_discriminator_real_acc= Accuracy(task = "binary")
        self.train_mixedup_discriminator_mixedup_acc= Accuracy(task = "binary")

        self.val_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.val_mixup_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.test_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.test_mixup_acc = Accuracy(task="multiclass", num_classes= N_WAY)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.val_mixup_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()

        self.initial_beta = initial_beta  # 초기 beta 값
        self.max_beta = max_beta     # 최대 beta 값
        self.epochs_to_max = epochs_to_max
        self.epochs_to_min = epochs_to_min
        self.initial_alpha = initial_alpha # 초기 alpha 값
        self.max_alpha = max_alpha      # 최대 alpha 값

    def pairwise_distances_logits(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        logits = -((a.unsqueeze(1).expand(n, m, -1) -
                    b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
        return logits
    def cosine_similarity(self, a, b):
    # Normalize vectors
        a_normalized = a / a.norm(dim=1, keepdim=True)
        b_normalized = b / b.norm(dim=1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.mm(a_normalized, b_normalized.t())
        
        return similarity
    
    def fast_adapt_distance(self, model, batch, ways, shot, mode, metric=None, device=None):
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
    
    def fast_adapt_cos(self, model, batch, ways, shot, mode, metric=None, device=None):
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
        logits = self.cosine_similarity(query, support)
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
        alpha = 0.95 + torch.rand(1).to(device) * 0.1
        mixed_imbedding = model.forward(mixed_data * alpha)
        mixed_imbedding = model.fault_classfier(mixed_imbedding)
        proto = mixed_imbedding.reshape(ways, 1, -1).mean(dim=1)
        query = embeddings[query_indices]
        labels = labels[query_indices].long()
        logits = self.cosine_similarity(query, proto)

        
        reconstructed_mixed_data = self.net.ae.forward(mixed_data.transpose(2,1))
        criterion = torch.nn.L1Loss()
        reconstruction_error = criterion(reconstructed_mixed_data, mixed_data.transpose(2,1))
        mixed_imbedding = model.forward(reconstructed_mixed_data.transpose(2,1))
        mixed_imbedding = model.fault_classfier(mixed_imbedding)
        proto = mixed_imbedding.reshape(ways, 1, -1).mean(dim=1)
        query = embeddings[query_indices]
        reconstructed_logits = self.cosine_similarity(query, proto)
        
        return logits, labels, reconstructed_logits, reconstruction_error
    
    def setup(self, stage=None):
        # 저장된 가중치 불러오기
        """
        if stage == 'fit' or stage is None:
            model_weights_path = '/home/geon/dev_geon/ml-testbed/src/models/components/ProtoNet_mixup_triplet_no_embedding.pth'
            self.net.load_state_dict(torch.load(model_weights_path))
        """
    def training_step(self, batch, batch_idx):

        current_epoch = self.current_epoch
        beta = self.initial_beta #+ (self.max_beta - self.initial_beta) * min(1, current_epoch / self.epochs_to_max)
        alpha = self.initial_alpha + (self.max_alpha - self.initial_alpha) * min(1, current_epoch / self.epochs_to_max)
        logits, labels = self.fast_adapt_cos(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        classification_loss = F.cross_entropy(logits, labels)

        mixedup_logits, mixedup_labels, reconstructed_logits, reconstruction_error  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        mixedup_classification_loss_1 = F.cross_entropy(mixedup_logits, mixedup_labels)
        mixedup_classification_loss_2 =  F.cross_entropy(reconstructed_logits, mixedup_labels)
        
        """
        mixedup_logits, mixedup_labels  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        mixedup_classification_loss_2 = F.cross_entropy(mixedup_logits, mixedup_labels)

        mixedup_logits, mixedup_labels  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        mixedup_classification_loss_3 = F.cross_entropy(mixedup_logits, mixedup_labels)

        mixedup_logits, mixedup_labels  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        mixedup_classification_loss_4 = F.cross_entropy(mixedup_logits, mixedup_labels)
        #fake_logit, fake_labels = self.fast_fake_adapt(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        #fake_classification_loss = F.cross_entropy(fake_logit, fake_labels)
        """
    
        anchor, positive, negative = self.net.prepare_triplet(batch)
        # Feature 추출
        anchor_feature = self.net.forward(anchor)
        positive_feature = self.net.forward(positive)    
        negative_feature = self.net.forward(negative)
        # Loss 계산
        triplet_loss = self.net.triplet_loss(anchor_feature, positive_feature, negative_feature)
        
        # 전체 손실
        total_loss = mixedup_classification_loss_2 + reconstruction_error + alpha * triplet_loss + (1-alpha)*(beta * classification_loss + (1-beta) * mixedup_classification_loss_1) # 0.25*mixedup_classification_loss_1 + 0.25*mixedup_classification_loss_2 + 0.25*mixedup_classification_loss_3 + 0.25*mixedup_classification_loss_4
            

        self.train_loss(total_loss)
        self.train_acc(logits, labels)
        self.train_mixed_up_acc(mixedup_logits, mixedup_labels)
        self.train_reconst_mixed_up_acc(reconstructed_logits, mixedup_labels)
        #self.train_fake_acc(fake_logit, fake_labels)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train_fake/acc', self.train_fake_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mixup/acc', self.train_mixed_up_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_reconst_mixup/acc', self.train_reconst_mixed_up_acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):

        logits, labels = self.fast_adapt_cos(self.net, batch, self.N_WAY, self.K_SHOT, mode = "val")
        classification_loss = F.cross_entropy(logits, labels)
        
        #mixedup_logits, mixedup_labels  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "val")
 

        self.val_loss(classification_loss)
        self.val_acc(logits, labels)
        #self.val_mixup_acc(mixedup_logits, mixedup_labels)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val_mixup/acc', self.val_mixup_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        mixup_acc = self.val_mixup_acc.compute()
        self.val_acc_best(acc)
        self.val_mixup_acc_best(mixup_acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True) 
        #self.log("val_mixup/acc_best", self.val_mixup_acc_best.compute(), sync_dist=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):

        logits, labels = self.fast_adapt_cos(self.net, batch, self.N_WAY, self.K_SHOT, mode = "test")
        torch.save(self.net.state_dict(), '/home/geon/dev_geon/ml-testbed/src/models/components/ProtoNet_mixup_triplet_no_embedding.pth')
        #mixedup_logits, mixedup_labels  = self.fast_adapt_mixedup_data(self.net, batch, self.N_WAY, self.K_SHOT, mode = "test")
    
        classification_loss = F.cross_entropy(logits, labels)
        self.test_loss(classification_loss)
        self.test_acc(logits, labels)
        #self.test_mixup_acc(mixedup_logits, mixedup_labels)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('test_mixup/acc', self.test_mixup_acc, on_step=False, on_epoch=True, prog_bar=True)

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