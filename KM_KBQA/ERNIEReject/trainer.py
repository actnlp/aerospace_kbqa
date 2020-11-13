"""
A trainer class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from . import torch_utils

from .ERNIE import BasicClassifier

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename,map_location=torch.device('cpu'))
        except BaseException as e:
            print(e)
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

# 0: tokens, 1: mask_s, 2: label
def unpack_batch(batch, cuda):
    inputs, label = batch[0:2], batch[2]
    if cuda:
        inputs = [Variable(i.cuda()) for i in inputs]
        label = Variable(label.cuda())
    else:
        inputs = [Variable(i) for i in inputs]
        label = Variable(label)
    return inputs, label

# 0: tokens, 1: mask_s, 2: label
class MyTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = BasicClassifier(opt)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        
        return loss.item(), acc

    def predict(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        self.model.eval()
        logits = self.model(inputs)
        # loss
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        return loss.item(), acc, predictions, label.data.cpu().numpy().tolist()

    def infer(self, batch,top_n):
        self.opt['cuda']=False
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        self.model.eval()
        logits = self.model(inputs)
        pred_complete = logits.data.cpu().numpy()
        preds_top = []
        for i in range(len(pred_complete)):
            pred_temp = [[j, item] for j, item in enumerate(pred_complete[i])]  # [label,logits]
            pred_temp.sort(key=lambda x: x[1], reverse=True)
            pred_top = pred_temp[0:top_n]
            preds_top.append(pred_top)
        # 取softmax
        preds_top = {item[0]: item[1] for i, item in enumerate(preds_top[0])}
        return preds_top
