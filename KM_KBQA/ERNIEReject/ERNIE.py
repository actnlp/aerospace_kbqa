import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_pretrained_bert import BertModel
import os
from . import torch_utils
from ..config import config

class BasicClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # ERNIE model
        opt['ERNIE_dir'] = config.ERNIE_path
        self.ERNIE = BertModel.from_pretrained(opt['ERNIE_dir'])
        # for param in self.ERNIE.parameters():
        #     param.requires_grad = True
        # dropout
        self.input_dropout = nn.Dropout(opt['input_dropout'])
        # classifier
        label2id = {"KBQA":0,'CQA':1}
        self.classifier = nn.Linear(opt['emb_dim'], len(label2id))
        
    # 0: tokens, 1: mask_s
    def forward(self, inputs):
        # unpack inputs 
        tokens, mask_s = inputs
        encoder_out, pooled = self.ERNIE(tokens, attention_mask=mask_s, output_all_encoded_layers=False)
        pooled = self.input_dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
