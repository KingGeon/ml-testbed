from typing import List, Optional, Tuple
import numpy as np
import pyrootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig
import time
import learn2learn as l2l
from models.components.conv_lstm_classifier import CONV_LSTM_Classifier
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.aihub_vibration_meta_taskmodule import Motor_vibration_TaskModule
from src.meta_learning.meta_utils import accuracy,pairwise_distances_logits,fast_adapt_maml, fast_adapt_proto, fast_adapt_maml_proto


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

    aihub_motor_vibration = Motor_vibration_TaskModule(ways = ways, shots = shots)
    aihub_motor_vibration.setup()
    train_tasks, valid_tasks, test_tasks = aihub_motor_vibration.make_tasks()
    
    pn_test_acc_list = []
    features = CONV_LSTM_Classifier()
        # for p in  features.parameters():
        #     print(p.shape)
    #features.load_state_dict(torch.load("../best_rca.pth"))
    features.to(device)
    head = torch.nn.Linear(5, ways)
    head = l2l.algorithms.MAML(head, lr=fast_lr)
    head.to(device)
    max_test_accuracy = 0
        # Setup optimization
        
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
            batch = train_tasks.sample()

            proto_loss, acc = fast_adapt_proto(features,
                                    batch,
                                    ways,
                                    shots,
                                    metric=pairwise_distances_logits,
                                    device=device)
            evaluation_error  = proto_loss 
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += acc.item() 

                # Compute meta-validation loss
            batch = valid_tasks.sample()
        
            proto_loss, acc = fast_adapt_proto(features,
                                    batch,
                                    ways,
                                    shots,
                                    metric=pairwise_distances_logits,
                                    device=device)
            evaluation_error  = proto_loss + evaluation_error
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += acc.item()

                # Compute meta-testing loss
            batch = test_tasks.sample()
            
            proto_loss, acc = fast_adapt_proto(features,
                                    batch,
                                    ways,
                                    shots,
                                    metric=pairwise_distances_logits,
                                    device=device)
            evaluation_error  = proto_loss + evaluation_error
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += acc.item()
            
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
        pn_test_acc_list.append(meta_test_accuracy / meta_bsz)
        if max_test_accuracy < meta_test_accuracy / meta_bsz:
            max_test_accuracy = meta_test_accuracy / meta_bsz
            # Average the accumulated gradients and optimize
                
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
        print(max_test_accuracy)



if __name__ == "__main__":
    main()
