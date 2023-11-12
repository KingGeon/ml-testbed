from typing import List, Optional, Tuple
import numpy as np
import pyrootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig
import time
import learn2learn as l2l
from models.components.conv_lstm_classifier_no_dropout import CONV_LSTM_Classifier
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.aihub_vibration_meta_taskmodule import Motor_vibration_TaskModule
from src.meta_learning.meta_utils import accuracy,pairwise_distances_logits,fast_adapt_maml, fast_adapt_proto, fast_adapt_maml_proto


def main():
    ways=4
    shots=4
    meta_head_lr=0.001
    meta_feature_lr=0.001
    fast_lr=0.1
    reg_lambda=0
    adapt_steps=5
    meta_bsz=32
    iters=500
    device = "cuda"

    cwru = Motor_vibration_TaskModule(ways = ways, shots = shots)
    cwru.prepare_data()
    cwru.setup()
    train_tasks, valid_tasks, test_tasks = cwru.make_tasks()
    
    features = CONV_LSTM_Classifier()
    # for p in  features.parameters():
    #     print(p.shape)
    #features.load_state_dict(torch.load("../best_rca.pth"))
    features.to(device)
    head = torch.nn.Linear(5, ways)
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
