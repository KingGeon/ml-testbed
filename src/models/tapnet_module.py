import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from lightning import LightningModule
import torch.optim as optim
import numpy as np

def nullspace_gpu(A, tol=1e-13):
    A = torch.atleast_2d(A)
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

class TapNetModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 N_WAY: int = 4,
                 K_SHOT: int = 4):
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        self.output_dimension = net.output_size
        self.projection_layer = nn.Linear(net.output_size, self.N_WAY, bias=False)

        self.train_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.val_acc = Accuracy(task="multiclass", num_classes= N_WAY)
        self.test_acc = Accuracy(task="multiclass", num_classes= N_WAY)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()

    def fast_adapt_tapnet(self, model, batch, ways, shot, mode, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.projection_layer.to(device)

        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)


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

        query_set = embeddings[query_indices]
        query_labels = labels[query_indices].long()
        batchsize_q = len(query_set)
        pow_avg = self.compute_power_avg_phi(prototype)
        phi_ind = [int(ind) for ind in self.select_phi(prototype, pow_avg)]
        # Compute the projection space
        M = self.Projection_Space(prototype, batchsize_q, self.N_WAY, mode, phi_ind=phi_ind)
    
        intermediate = torch.bmm(M.transpose(1, 2), query_set.unsqueeze(-1))
        # 두 번째 배치 매트릭스 곱셈
        r_t = torch.bmm(M, intermediate)
        # 최종 결과의 형태를 변경
        r_t = r_t.view(batchsize_q, -1)
        pow_t = self.compute_power(batchsize_q, query_set, M, self.N_WAY, mode, phi_ind=phi_ind)
        pred = self.predict(query_labels, r_t, pow_t, phi_ind=phi_ind)
        loss = self.compute_loss(query_labels, r_t, pow_t, self.N_WAY)


        return pred, query_labels, loss

    def select_phi(self, prototype, avg_pow):
        u_avg = 2 * self.projection_layer(prototype)  
        u_avg = u_avg - avg_pow

        u_avg_ind = torch.argsort(u_avg, dim=1)

        phi_ind = torch.zeros(self.N_WAY, dtype=torch.int64, device=u_avg.device)  # Ensure phi_ind is on the same device as u_avg_ind

        for i in range(self.N_WAY):
            if i == 0:
                phi_ind[i] = u_avg_ind[i, self.N_WAY - 1]
            else:
                k = self.N_WAY - 1
                while u_avg_ind[i, k] in phi_ind[:i]:
                    k -= 1
                phi_ind[i] = u_avg_ind[i, k]

        return phi_ind.tolist()

    def compute_power(self, batchsize, key, M, nb_class, train=True, phi_ind=None):
        if train:
            Phi_out = self.projection_layer.weight  # PyTorch에서의 가중치 접근
        else:
            Phi_data = self.projection_layer.weight.detach()  # .detach() 사용
            Phi_out = Phi_data[phi_ind, :]

        Phi_out_batch = Phi_out.expand(batchsize, nb_class, self.output_dimension)  # 브로드캐스팅
        PhiM = torch.bmm(Phi_out_batch, M)  # 배치 매트릭스 곱셈
        PhiMs = torch.sum(PhiM * PhiM, dim=2)  # 합계 연산

        key_t = key.view(batchsize, 1, self.output_dimension)  # 형태 변경
        keyM = torch.bmm(key_t, M)
        keyMs = torch.sum(keyM * keyM, dim=2)
        keyMs = keyMs.expand(batchsize, nb_class)  # 브로드캐스팅

        pow_t = PhiMs + keyMs

        return pow_t
    
    def compute_power_avg_phi(self, average_key, train=False):
        avg_pow = torch.sum(average_key * average_key, dim=1)
        Phi = self.projection_layer.weight  # PyTorch에서의 가중치 접근
        Phis = torch.sum(Phi * Phi, dim=1)

        avg_pow_bd = avg_pow.view(len(avg_pow), 1).expand(len(avg_pow), len(Phis))  # 브로드캐스팅 및 형태 변경
        wzs_bd = Phis.view(1, len(Phis)).expand(len(avg_pow), len(Phis))  # 브로드캐스팅 및 형태 변경

        pow_avg = avg_pow_bd + wzs_bd

        return pow_avg
    
    def compute_loss(self, t_data, r_t, pow_t, mode="train"):
        t = torch.tensor(t_data, dtype=torch.int64)  # Ensure the data type is correct
        u = 2 * self.projection_layer(r_t) - pow_t
        return F.cross_entropy(u, t)

    def predict(self, t_data, r_t, pow_t, phi_ind=None):
        ro = 2 * self.projection_layer(r_t)  # 여기서 self.projection_layer는 PyTorch의 레이어로 정의되어야 함
        ro_t = ro.detach()[:, phi_ind]  # .detach() 사용
        u = ro_t - pow_t

        t_est = torch.argmax(torch.softmax(u, dim=1), dim=1)  # PyTorch의 argmax와 softmax 사용

        return t_est

    def Projection_Space(self, prototype, batchsize, nb_class, mode="train", phi_ind=None):
        device = prototype.device
        c_t = prototype.to(device)  # Ensure prototype is on the correct device
        eps = 1e-6

        if mode == "train":
            Phi_tmp = self.projection_layer.weight
        else:
            Phi_data = self.projection_layer.weight.data
            Phi_tmp = Phi_data[phi_ind,:].clone().detach()  # 수정된 부분

        Phi_sum = Phi_tmp[0].clone().detach()  # 첫 번째 요소로 Phi_sum 초기화
        for i in range(1, nb_class):
            Phi_sum += Phi_tmp[i]
        Phi = (nb_class * Phi_tmp) - Phi_sum.expand(nb_class, self.output_dimension)
        power_Phi = torch.sqrt(torch.sum(Phi * Phi, axis=1))
        power_Phi = power_Phi.expand(self.output_dimension, nb_class).permute(1, 0)
        
        Phi = Phi / (power_Phi + eps)

        power_c = torch.sqrt(torch.sum(c_t * c_t, axis=1))
        power_c = power_c.expand(self.output_dimension, nb_class).permute(1, 0)
        c_tmp = c_t / (power_c + eps)

        null = Phi - c_tmp
        M = nullspace_gpu(null.detach())  # 또는 null.detach()를 사용하는 것을 고려할 수 있음
        M = M.expand(batchsize, self.output_dimension, self.output_dimension - nb_class)
        return M


    
    def training_step(self, batch, batch_idx):
        pred, labels, loss = self.fast_adapt_tapnet(self.net, batch, self.N_WAY, self.K_SHOT, mode = "train")
        self.train_loss(loss)
        self.train_acc(pred, labels)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        pred, labels, loss = self.fast_adapt_tapnet(self.net, batch, self.N_WAY, self.K_SHOT, mode = "val")
        self.val_loss(loss)
        self.val_acc(pred, labels)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True), 
        
    def test_step(self, batch, batch_idx):
        pred, labels, loss = self.fast_adapt_tapnet(self.net, batch, self.N_WAY, self.K_SHOT, mode = "test")
        self.test_loss(loss)
        self.test_acc(pred, labels)
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
