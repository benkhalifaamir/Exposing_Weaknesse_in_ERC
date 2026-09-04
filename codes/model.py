import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.utils import Bunch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class ERC_model(nn.Module):
    def __init__(self, model_type, clsNum, last, work_dir):
        super(ERC_model, self).__init__()
        self.gpu = True
        self.last = last
        self.work_dir = work_dir
        """Model Setting"""        
        
        model_path = model_type 
        self.model = RobertaModel.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.pad_value = self.tokenizer.pad_token_id
            
        condition_token = ['<s1>', '<s2>', '<s3>'] 
        special_tokens = {'additional_special_tokens': condition_token}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)
    def device(self):
        return self.model.device
    
    def hidden(self):
        return self.hiddenDim
        
    def forward(self, batch_input_tokens):

        batch_size, max_len = batch_input_tokens.shape[0], batch_input_tokens.shape[-1]
        mask = 1 - (batch_input_tokens == (self.pad_value)).long()
        
        output = self.model(
            input_ids = batch_input_tokens, 
            attention_mask = mask,
            output_hidden_states = True,
            )
        
        batch_context_output = output.last_hidden_state[:,0,:]
        
        context_logit = self.W(batch_context_output)
        
        return context_logit, batch_context_output, output.hidden_states
