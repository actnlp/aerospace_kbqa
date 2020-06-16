import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

pattern = re.compile(r'([\u4e00-\u9fa5])')


def cut_char(s):
    out = [p for p in pattern.split(s) if p != '']
    return out


class DynamicLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicLSTM, self).__init__()
        kwargs['batch_first'] = True
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, input_seq, lengths):
        total_len = input_seq.size(1)
        sorted_lengths, idx = torch.sort(lengths, dim=0, descending=True)
        _, un_idx = torch.sort(idx, dim=0)
        input_seq = input_seq[idx]
        packed_seq = pack_padded_sequence(
            input_seq, sorted_lengths, batch_first=True)
        output, _ = self.lstm(packed_seq)
        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=total_len)
        output = output[un_idx]
        return output


class LSTMEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 output_dim=2):
        super(LSTMEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self.lstm = DynamicLSTM(
            embedding_dim, embedding_dim//2, bidirectional=True)
        self._output_dim = output_dim
        self.attn = nn.Linear(embedding_dim, 1)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, X, lengths):
        '''
            X : input [batch_size, word_num, vec_dim]
        '''
        output = self.lstm(X, lengths)
        a = self.attn(output)
        output = torch.sum(a*output, dim=1)
        output = self.fc(output)
        return output


class CNNEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 filter_num,
                 ngrams=list(range(1, 7)),
                 output_dim=None):
        super(CNNEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._filter_num = filter_num
        self._ngrams = ngrams
        self._output_dim = output_dim
        self._conv_layers = nn.ModuleList([nn.Conv1d(self._embedding_dim, self._filter_num, ngram_size, padding=ngram_size-1)
                                           for ngram_size in self._ngrams])
        maxpool_output_dim = self._filter_num*len(self._ngrams)
        self.dropout = nn.Dropout(0.4)
        if self._output_dim:
            self.projection_layer = nn.Linear(
                maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def forward(self, X, lengths=None):
        '''
            X : input [batch_size, word_num, vec_dim]
        '''
        X = X.transpose(1, 2)
        conv_outputs = tuple(F.relu(conv(X))
                             for conv in self._conv_layers)
        pool_outputs = tuple(conv_output.max(dim=2)[0]
                             for conv_output in conv_outputs)
        pool_output = torch.cat(pool_outputs, dim=1)
        # pool_output = self.dropout(pool_output)

        if self.projection_layer:
            result = self.projection_layer(pool_output)
        else:
            result = pool_output
        return result


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        # print('block shape:', x.shape)
        return x


class DPCNN(nn.Module):
    """
    DPCNN model, 3
    1. region embedding: using TetxCNN to generte
    2. two 3 conv(padding) block
    3. maxpool->3 conv->3 conv with resnet block(padding) feature map: len/2
    """

    # max_features, opt.EMBEDDING_DIM, opt.SENT_LEN, embedding_matrix):
    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 filter_num,
                 n_class,
                 sent_len=32):
        super(DPCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.filter_num = filter_num
        self.n_class = n_class
        self.sent_len = sent_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # region embedding
        self.region_embd = nn.Sequential(
            nn.Conv1d(embedding_dim, filter_num,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=filter_num),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=filter_num),
            nn.ReLU(),
            nn.Conv1d(filter_num, filter_num,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=filter_num),
            nn.ReLU(),
            nn.Conv1d(filter_num, filter_num,
                      kernel_size=3, padding=1),
        )

        resnet_block_list = []
        while (sent_len > 2):
            resnet_block_list.append(ResnetBlock(filter_num))
            sent_len = sent_len // 2
        self.final_size = filter_num*sent_len
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(
            nn.Linear(self.filter_num, n_class),
            nn.BatchNorm1d(n_class),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(n_class, n_class)
        )

    def get_args(self):
        return {'embedding_dim': self.embedding_dim,
                'vocab_size': self.vocab_size,
                'filter_num': self.filter_num,
                'n_class': self.n_class,
                'sent_len': self.sent_len}

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embd(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        # x = x.permute(0, 2, 1)
        x = x.max(dim=2)[0]
        # print(x.shape)
        # print(x.shape, self.final_size)
        # x = x.contiguous().view(-1, self.filter_num)
        out = self.fc(x)
        return out

    def _reset_params(self):  # reset model parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'embedding' in name:  # treat embedding matrices as special cases
                    # use xavier_uniform to initialize embedding matrices
                    weight = torch.nn.init.xavier_uniform_(
                        torch.zeros_like(param))
                    # the vector corresponding to padding index shuold be zero
                    weight[0] = torch.tensor(
                        0, dtype=param.dtype, device=param.device)
                    setattr(param, 'data', weight)  # update embedding matrix
                else:
                    if len(param.shape) > 1:
                        # use xavier_uniform to initialize weight matrices
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        stdv = 1. / math.sqrt(param.size(0))
                        # use uniform to initialize bias vectors
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv)


class QCLS(nn.Module):
    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 encoder_type='cnn'):
        super(QCLS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_type = encoder_type
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(embedding_dim, 32, output_dim=2)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(embedding_dim, output_dim=2)

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

    def get_args(self):
        return {'embedding_dim': self.embedding_dim,
                'vocab_size': self.vocab_size,
                'encoder_type': self.encoder_type}

    def forward(self, x, lengths=None):
        e = self.embedding(x)
        output = self.encoder(e, lengths)
        return output

    def _reset_params(self):  # reset model parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'embedding' in name:  # treat embedding matrices as special cases
                    # use xavier_uniform to initialize embedding matrices
                    weight = torch.nn.init.xavier_uniform_(
                        torch.zeros_like(param))
                    # the vector corresponding to padding index shuold be zero
                    weight[0] = torch.tensor(
                        0, dtype=param.dtype, device=param.device)
                    setattr(param, 'data', weight)  # update embedding matrix
                else:
                    if len(param.shape) > 1:
                        # use xavier_uniform to initialize weight matrices
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        stdv = 1. / math.sqrt(param.size(0))
                        # use uniform to initialize bias vectors
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv)


class QCLSWrapper():
    def __init__(self, model, tokenizer):
        self.model = model.cpu()
        self.tokenizer = tokenizer
        self.model.eval()

    def dump(self, file):
        torch.save({
            'model': self.model.state_dict(),
            'args': self.model.get_args(),
            'index2word': self.tokenizer.index2word
        }, file)

    @staticmethod
    def from_pretrained(file, core='qcls'):
        states = torch.load(file)
        if core == 'qcls':
            model = QCLS(**states['args'])
        elif core == 'dpcnn':
            model = DPCNN(**states['args'])
        else:
            model = QCLS(**states['args'])
        model.load_state_dict(states['model'])
        tokenizer = Tokenizer(states['index2word'])
        return QCLSWrapper(model, tokenizer)

    def eval(self, text_batch):
        self.model.eval()
        with torch.no_grad():
            x, l = self.tokenizer.process(text_batch)
            pred = torch.softmax(self.model(x, l), dim=1).numpy()[:, 0]
            return pred


class Tokenizer():
    def __init__(self, index2word, cut=cut_char):
        self.index2word = index2word
        self.word2index = dict(zip(index2word, range(len(index2word))))
        self.cut = cut

    def tokenize(self, sent):
        return self.cut(sent)

    def convert2id(self, token_list):
        return [self.word2index.get(t, self.word2index['<unk>']) for t in token_list]

    def pad(self, seq_list):
        seq_list = list(seq_list)
        lengths = torch.tensor(list(map(len, seq_list)))
        tensors = list(map(torch.tensor, seq_list))
        tensors = pad_sequence(tensors, batch_first=True,
                               padding_value=self.word2index['<pad>'])
        return tensors, lengths

    def process(self, text_batch):
        tokens = map(self.tokenize, text_batch)
        idxs = map(self.convert2id, tokens)
        return self.pad(idxs)


if __name__ == '__main__':
    print(cut_char('机场哪里有KFC的饭店'))
