import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import pickle
import random
import time
import timeit
import warnings

class SupconLoss(nn.Module):
    def __init__(self, num_classes, temp,neu_value, args):
        super().__init__()
        self.temperature = temp
        self.num_classes = num_classes
        self.neu_value = neu_value
        self.eps = 1e-8
        self.args = args
        
    def score_func(self, x, y):
        return (1+F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def forward(self, reps, labels, augmentation=None, aug_label=None, decoupled=False):
        batch_size = reps.shape[0]
        
        if self.args.augment != '' and len(augmentation) >1 :
            concated_reps = torch.cat((reps, augmentation), 0)
            concated_labels = torch.cat((labels, aug_label), 0)
            concated_bsz = batch_size + augmentation.shape[0]
        else:
            concated_reps = reps
            concated_labels = labels
            concated_bsz = batch_size
        
        mask1 = concated_labels.unsqueeze(0).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask2 = concated_labels.unsqueeze(1).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        pos_mask = (mask1 == mask2).long()
        
        rep1 = concated_reps.unsqueeze(0).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        rep2 = concated_reps.unsqueeze(1).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(concated_reps.device)
        
        scores /= self.temperature
        
        scores -= torch.max(scores).item()
        
        
        neu = torch.full((concated_labels.shape[0], concated_labels.shape[0]), fill_value = self.neu_value).to(scores.device) 
        neu_mask = ( 1 - ((mask1 != neu)==(mask2 != neu)).long())
        neu_mask = neu_mask
        
        
        scores = torch.exp(scores)
        pos_scores = scores * (pos_mask * mask)
        if self.args.loss == 'neu':
            neg_scores = scores * (1 - pos_mask) + self.args.prob *(scores * neu_mask) 
        else:
            neg_scores = scores * (1 - pos_mask)
        probs = pos_scores.sum(-1)/(pos_scores.sum(-1) + neg_scores.sum(-1))
        probs /= (pos_mask * mask).sum(-1) + self.eps
        loss = - torch.log(probs + self.eps)
        loss_mask = (loss > 0.3).long()
        loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
            
        return loss
