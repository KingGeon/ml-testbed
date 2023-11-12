from typing import List, Optional, Tuple
import numpy as np
import lightning as L
import pyrootutils
import torch
import torch.nn as nn
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import learn2learn as l2l
from models.components.conv_lstm_classifier import CONV_LSTM_Classifier
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src import utils
from src.data.

log = utils.get_pylogger(__name__)

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt_maml(batch,
               learner,
               features,
               loss,
               adaptation_steps,
               shots,
               ways,
               device=None):

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data = features(data)
    

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def main():
    ways=4
    shots=4
    meta_head_lr=0.001
    meta_tail_lr = 0.001
    meta_feature_lr=0.001
    fast_lr=0.1
    reg_lambda=0
    adapt_steps=5
    meta_bsz=32
    iters=500
    cuda=1
    seed=42
    device = "cuda"

    cwru = CWRUDataModule(ways = ways, shots = shots)
    cwru.prepare_data()
    cwru.setup()
    train_tasks, valid_tasks, test_tasks = cwru.make_tasks()
    
    features = CONV_LSTM_Classifier()
    # for p in  features.parameters():
    #     print(p.shape)
    #features.load_state_dict(torch.load("../best_rca.pth"))
    features.to(device)
    head = torch.nn.Linear(4, ways)
    head = l2l.algorithms.MAML(head, lr=fast_lr)
    head.to(device)

        # Setup optimization
    all_parameters = list(features.parameters()) + list(head.parameters())
        
        # optimizer = torch.optim.Adam(all_parameters, lr=meta_lr)
        
        ## use different learning rates for w and theta
    optimizer = torch.optim.Adam([{'params': list(head.parameters()), 'lr': meta_head_lr},
        {'params': list(features.parameters()), 'lr': meta_feature_lr}])
        
        
    loss = nn.CrossEntropyLoss(reduction='mean')
        
    training_accuracy =  torch.ones(iters)
    test_accuracy =  torch.ones(iters)
    running_time = np.ones(iters)
    import time
    start_time = time.time()

    for iteration in range(iters):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
            
        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = head.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt_maml(batch,
                                                                learner,
                                                                features,
                                                                loss,
                                                                adapt_steps,
                                                                shots,
                                                                ways,
                                                                device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
            learner = head.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt_maml(batch,
                                                                learner,
                                                                features,
                                                                loss,
                                                                adapt_steps,
                                                                shots,
                                                                ways,
                                                                device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

                # Compute meta-testing loss
            learner = head.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt_maml(batch,
                                                                learner,
                                                                features,
                                                                loss,
                                                                adapt_steps,
                                                                shots,
                                                                ways,
                                                                device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
            
        training_accuracy[iteration] = meta_train_accuracy / meta_bsz
        test_accuracy[iteration] = meta_test_accuracy / meta_bsz

            # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        print('Meta Valid Error', meta_valid_error / meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
        print('Meta Test Error', meta_test_error / meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

            # Average the accumulated gradients and optimize
        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_bsz)
                
        # print('head')
        # for p in list(head.parameters()):
        #     print(torch.max(torch.abs(p.grad.data)))
                
        # print('feature')
        # for p in list(features.parameters()):
        #     print(torch.max(torch.abs(p.grad.data)))
            
        optimizer.step()
        end_time = time.time()
        running_time[iteration] = end_time - start_time
        print('total running time', end_time - start_time)



if __name__ == "__main__":
    main()
