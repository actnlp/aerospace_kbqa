import os
import pdb
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from transformers import BertModel, BertPreTrainedModel

from . import args
from .crf import CRF

class_report_path = os.path.join(args.log_path, 'class_report.log')


class MultiTaskKBQA(nn.Module):
    def __init__(self, config):
        super(MultiTaskKBQA, self).__init__()
        # self.bert = BertEncoder.from_pretrained(args.bert_model)
        self.bert = BertEncoder.from_pretrained('bert-base-chinese')
        self.crf = CRFEncoder(config)
        self.cnn_encoder_1 = nn.Conv1d(
            in_channels=args.crf_hidden_size, out_channels=args.bilstm_hidden_size, kernel_size=3, padding=1)
        self.cnn_encoder_2 = nn.Conv1d(
            in_channels=args.crf_hidden_size, out_channels=args.bilstm_hidden_size, kernel_size=5, padding=2)
        self.bilstm = BiLSTMEncoder(config)
        self.ent_mlp = MLP(config, mode='ent')
        self.rel_mlp = MLP(config, mode='rel')
        update_hidden_size = config.crf_hidden_size + \
            config.num_tag +\
            config.mlp_ent_output_size +\
            config.mlp_rel_output_size
        self.update_mlp = nn.Sequential(
            nn.Linear(update_hidden_size, update_hidden_size),
            nn.ReLU(),
            nn.Linear(update_hidden_size, config.bilstm_input_size)
        )
        # attention
        if args.use_soft_attention:
            self.ent_w = torch.nn.Parameter(torch.FloatTensor(
                args.bilstm_hidden_size * args.bilstm_num_layers, args.num_tag))
            self.rel_w = torch.nn.Parameter(torch.FloatTensor(
                args.bilstm_hidden_size * args.bilstm_num_layers, args.num_tag))
            nn.init.xavier_uniform_(self.ent_w)
            nn.init.xavier_uniform_(self.rel_w)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                batch_idx):
        bert_encode, bert_embedding = self.bert(
            input_ids, token_type_ids, attention_mask)

        i = 0
        while True:
            slot_filling_output = self.crf(bert_encode)
            merged_bilstm_input = bert_encode + bert_embedding
            cnn_1 = F.relu(self.cnn_encoder_1(
                merged_bilstm_input.transpose(1, 2)))
            cnn_2 = F.relu(self.cnn_encoder_1(
                merged_bilstm_input.transpose(1, 2)))
            # pack sequence
            token_length = torch.sum(attention_mask, dim=-1)
            padded_merged = torch.cat([cnn_1, cnn_2], dim=1).transpose(1, 2)

            ent_bilstm_output, rel_bilstm_output, bilstm_output = [], [], []
            if args.use_soft_attention:
                # padded_merged, padded_length = padded_merged[reversed_index], padded_length[reversed_index]
                # add attention mechanism
                ent_atten_1 = torch.matmul(padded_merged, self.ent_w)
                ent_atten_2 = slot_filling_output.transpose(1, 2)
                ent_atten = torch.matmul(ent_atten_1, ent_atten_2)
                # pdb.set_trace()
                for idx, length in enumerate(token_length):
                    cur_ent_atten = F.softmax(
                        ent_atten[idx, :length, :length] / args.attention_T, dim=-1)
                    cur_slot_filling = slot_filling_output[idx, :length, 0] + slot_filling_output[idx,
                                                                                                  :length, 1] + slot_filling_output[idx, :length, 6] + slot_filling_output[idx, :length, 7]
                    cur_slot_filling = F.softmax(
                        cur_slot_filling / args.attention_T, dim=-1)
                    cur_ent_atten = cur_ent_atten * \
                        cur_slot_filling.unsqueeze(0)
                    cur_ent_atten = torch.sum(
                        cur_ent_atten, dim=-1, keepdim=True)
                    cur_ent_atten = F.softmax(
                        cur_ent_atten / args.attention_T, dim=0).transpose(0, 1)
                    cur_padded_merged = padded_merged[idx, :length, :]
                    cur_ent_atten_res = torch.matmul(
                        cur_ent_atten, cur_padded_merged)
                    ent_bilstm_output.append(cur_ent_atten_res.squeeze())

                ent_bilstm_output = torch.stack(ent_bilstm_output, dim=0)

                # add attention mechanism
                rel_atten_1 = torch.matmul(padded_merged, self.rel_w)
                rel_atten_2 = slot_filling_output.transpose(1, 2)
                rel_atten = torch.matmul(rel_atten_1, rel_atten_2)

                for idx, length in enumerate(token_length):
                    cur_rel_atten = F.softmax(
                        rel_atten[idx, :length, :length] / args.attention_T, dim=-1)
                    cur_slot_filling = slot_filling_output[idx,
                                                           :length, 2] + slot_filling_output[idx, :length, 3]
                    cur_slot_filling = F.softmax(
                        cur_slot_filling / args.attention_T, dim=-1)
                    cur_rel_atten = cur_rel_atten * \
                        cur_slot_filling.unsqueeze(0)

                    cur_rel_atten = torch.sum(
                        cur_rel_atten, dim=-1, keepdim=True)
                    cur_rel_atten = F.softmax(
                        cur_rel_atten / args.attention_T, dim=0).transpose(0, 1)
                    cur_padded_merged = padded_merged[idx, :length, :]
                    cur_rel_atten_res = torch.matmul(
                        cur_rel_atten, cur_padded_merged)
                    rel_bilstm_output.append(cur_rel_atten_res.squeeze())

                rel_bilstm_output = torch.stack(rel_bilstm_output, dim=0)

                ent_cls_output = self.ent_mlp(ent_bilstm_output)
                rel_cls_output = self.rel_mlp(rel_bilstm_output)

            else:
                cnn_averaged_output = []
                for idx, length in enumerate(token_length):
                    cnn_averaged_output.append(torch.mean(
                        padded_merged[idx, :length, :], dim=0))
                cnn_averaged_output = torch.stack(cnn_averaged_output, dim=0)

                ent_cls_output = self.ent_mlp(cnn_averaged_output)
                rel_cls_output = self.rel_mlp(cnn_averaged_output)

            if i < args.interaction:  # and batch_idx % 5 == 0 and batch_idx > 0:
                update_input = torch.cat(
                    (bert_encode,
                     slot_filling_output,
                        ent_cls_output[:, None,
                                       :].expand(-1, bert_encode.size(1), -1),
                        rel_cls_output[:, None, :].expand(-1, bert_encode.size(1), -1)),
                    dim=-1)
                bert_encode = self.update_mlp(update_input)  # + bert_encode
                i += 1
            else:
                break
        return slot_filling_output, ent_cls_output, rel_cls_output

    def loss_fn(self, slot_filling_output, ent_mlp_output, rel_mlp_output, output_mask, tags, ent_tags, rel_tags):
        slot_filling_loss = self.crf.loss_fn(
            slot_filling_output, output_mask, tags)

        ent_mlp_loss = self.ent_mlp.loss(ent_mlp_output, ent_tags)
        rel_mlp_loss = self.rel_mlp.loss(rel_mlp_output, rel_tags)
        return slot_filling_loss, ent_mlp_loss, rel_mlp_loss

    def predict_slot_filling(self, slot_filling_output, output_mask):
        return self.crf.predict(slot_filling_output, output_mask)

    def predict_ent_classify(self, ent_mlp_output):
        return self.ent_mlp.predict(ent_mlp_output)

    def predict_rel_classify(self, rel_mlp_output, ent_top3):
        return self.rel_mlp.predict(rel_mlp_output)

    def acc_f1(self, slot_pred, slot_true, ent_cls_pred, ent_cls_true, rel_cls_pred, rel_cls_true):
        slot_acc, slot_f1 = self.crf.acc_f1(slot_pred, slot_true)
        ent_cls_acc, ent_cls_f1 = self.ent_mlp.acc_f1(
            ent_cls_pred, ent_cls_true)
        rel_cls_acc, rel_cls_f1 = self.rel_mlp.acc_f1(
            rel_cls_pred, rel_cls_true)

        return slot_acc, slot_f1, ent_cls_acc, ent_cls_f1, rel_cls_acc, rel_cls_f1

    def overall_f1(self, ent_cls_top3, ent_cls_true, rel_cls_top3, rel_cls_true):
        count_sum, correct_top1, correct_top3 = ent_cls_true.shape[0], 0, 0
        is_true_top1, is_true_top3 = [], []
        # compute top1 f1
        ent_cls_predicts, rel_cls_predicts = ent_cls_top3[:,
                                                          0], rel_cls_top3[:, 0]
        for i in range(count_sum):
            if ent_cls_predicts[i] == ent_cls_true[i] and rel_cls_predicts[i] == rel_cls_true[i]:
                correct_top1 += 1
                is_true_top1.append(1)
            else:
                is_true_top1.append(0)
        top1_f1 = correct_top1 / count_sum
        # compute top3 f1
        for i in range(count_sum):
            if ent_cls_true[i] in ent_cls_top3[i] and rel_cls_predicts[i] == rel_cls_true[i]:
                correct_top3 += 1
                is_true_top3.append(1)
            else:
                is_true_top3.append(0)
        top3_f1 = correct_top3 / count_sum

        return top1_f1, top3_f1, is_true_top1, is_true_top3

    def class_report(self, slot_pred, slot_true, ent_cls_pred, ent_cls_true, rel_cls_pred, rel_cls_true, epoch):
        self.crf.class_report(slot_pred, slot_true, epoch)
        self.ent_mlp.class_report(ent_cls_pred, ent_cls_true, epoch)
        self.rel_mlp.class_report(rel_cls_pred, rel_cls_true, epoch)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                label_id=None):
                # output_all_encoded_layers=False):
        bert_encode, _ = self.bert(input_ids, token_type_ids, attention_mask,)
                                #    output_all_encoded_layers=output_all_encoded_layers)
        bert_embeddings = self.bert.embeddings(input_ids, token_type_ids)

        return bert_encode, bert_embeddings


class CRFEncoder(nn.Module):
    def __init__(self, config):
        super(CRFEncoder, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.crf_hidden_size, config.num_tag)

        self.crf_layer = CRF(config.num_tag)

    def forward(self, bert_encode):
        output = self.classifier(bert_encode)
        output_dropped = self.dropout(output)
        return output_dropped

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf_layer.negative_log_loss(bert_encode, output_mask, tags)

        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf_layer.get_batch_best_path(bert_encode, output_mask)
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        try:
            f1 = f1_score(y_true, y_pred, average="weighted")
        except:
            pdb.set_trace()
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true, epoch):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(
            y_true, y_pred, target_names=args.labels)
        print('\n\nclassify_report:\n', classify_report)
        with open(class_report_path, 'a+') as f:
            print('\nEPOCH: %d slot_classify_report:\n' % epoch, file=f)
            print(classify_report, file=f)


class BiLSTMEncoder(nn.Module):
    def __init__(self, config, batch_first=True, bi_flag=True):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = config.bilstm_input_size
        self.hidden_size = config.bilstm_hidden_size
        self.num_layers = config.bilstm_num_layers
        self.dropout = config.bilstm_dropout
        self.bi_flag = bi_flag

        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                              batch_first=batch_first, dropout=self.dropout, bidirectional=bi_flag)

    def forward(self, input):
        output, _ = self.bilstm(input)
        return output

    def init_weights(self, batch_size):
        direction = 2 if self.bi_flag else 1
        h0 = torch.empty(self.num_layers * direction, batch_size,
                         args.bilstm_hidden_size).to(args.device)
        c0 = torch.empty(self.num_layers * direction, batch_size,
                         args.bilstm_hidden_size).to(args.device)
        torch.nn.init.orthogonal_(h0)
        torch.nn.init.orthogonal_(c0)
        return h0, c0


class MLP(nn.Module):
    def __init__(self, config, mode):
        super(MLP, self).__init__()
        self.input_size = config.mlp_input_size
        self.hidden_size = config.mlp_hidden_size
        if mode == 'ent':
            self.output_size = config.mlp_ent_output_size
        elif mode == 'rel':
            self.output_size = config.mlp_rel_output_size
        else:
            raise NotImplementedError("Invalid mode %s" % mode)
        self.mode = mode

        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        output = self.mlp(input)
        return output

    def loss_fn(self, mlp_output, labels):
        loss = self.loss(mlp_output, labels)
        return loss

    def predict(self, mlp_output):
        mlp_output = torch.softmax(mlp_output, dim=-1)
        # compute top3 answers
        _, top3_ans = torch.topk(mlp_output, 3, dim=-1)
        predicts = torch.argmax(mlp_output, dim=-1).detach()
        return predicts, top3_ans

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="weighted")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true, epoch):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        target_names = args.REL_LABELS if self.mode == 'rel' else args.ENT_LABELS
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
        with open(class_report_path, 'a+') as f:
            print('\nEPOCH: %d %s_classify_report:\n' %
                  (epoch, self.mode), file=f)
            print(classify_report, file=f)
