import copy
import pdb
import time
import warnings

import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from . import args
from .optimization import BertAdam
from .utils.eval_res_util import aeas_write_eval_res, write_eval_res
from .utils.Logginger import init_logger
from .utils.model_util import save_model
from .utils.plot_util import (loss_acc_f1_plot, loss_acc_plot,
                              one_loss_acc_f1_plot)

logger = init_logger("torch", logging_path=args.log_path)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

warnings.filterwarnings('ignore')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def multitask_fit(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, k, verbose=1):
    # ------------------tensorboardX 可视化----------------------
    writer = SummaryWriter(args.tensorboard_path)
    # ------------------判断CUDA模式----------------------
    device = torch.device(
        args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'bert' in n],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'bert' in n],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not 'bert' in n],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not 'bert' in n],
         'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    # ---------------------GPU半精度fp16-----------------------------
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)
    # ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    # ---------------------模型初始化----------------------
    if args.fp16:
        model.half()

    model.to(device)

    train_losses = []
    eval_losses = []
    eval_slot_accuracy, eval_ent_accuracy, eval_rel_accuracy = [], [], []
    eval_slot_f1_score, eval_ent_f1_score, eval_rel_f1_score = [], [], []
    eval_overall_top1_f1, eval_overall_top3_f1 = [], []

    history = {
        "train_loss": train_losses,
        "eval_loss": eval_losses,
        "eval_slot_acc": eval_slot_accuracy,
        "eval_ent_acc": eval_ent_accuracy,
        "eval_rel_acc": eval_rel_accuracy,
        "eval_slot_f1": eval_slot_f1_score,
        "eval_ent_f1": eval_ent_f1_score,
        "eval_rel_f1": eval_rel_f1_score,
        "eval_overall_top1": eval_overall_top1_f1,
        "eval_overall_top3": eval_overall_top3_f1
    }

    # ------------------------训练------------------------------
    best_f1 = 0
    no_increment = 0
    start = time.time()
    global_step = 0
    for e in range(num_epoch):
        model.train()
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(device) for t in batch)
            tr_idx, input_ids, input_mask, segment_ids, output_mask, label_ids, ent_ids, rel_ids = batch
            # print("input_id", input_ids)
            # print("input_mask", input_mask)
            # print("segment_id", segment_ids)
            slot_filling_output, ent_cls_output, rel_cls_output = model(
                input_ids, segment_ids, input_mask, batch_idx=step)
            slot_filling_output, ent_cls_output, rel_cls_output = slot_filling_output.cpu(
            ), ent_cls_output.cpu(), rel_cls_output.cpu()

            try:
                tr_slot_loss, tr_ent_loss, tr_rel_loss = model.loss_fn(slot_filling_output, ent_cls_output, rel_cls_output,
                                                                       output_mask, label_ids, ent_ids.cpu(), rel_ids.cpu())
            except:
                pdb.set_trace()
            train_loss = tr_slot_loss + tr_ent_loss + tr_rel_loss

            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(train_loss)
            else:
                train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)

            # hook gradients
            # for name, parms in model.named_parameters():
            #     if 'ent_mlp.mlp' in name:
            #         print('-->name:', name, ' -->grad_requirs:', parms.requires_grad, \
            #             '\n-->node_value:', parms)
            #         # ' \n-->grad_value:',parms.grad,
            #         print(parms.grad.shape)
            # time.sleep(2)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * \
                    warmup_linear(global_step / t_total,
                                  args.warmup_proportion)
                # only apply to bert layers
                for param_group in optimizer.param_groups[:2]:
                    param_group['lr'] = lr_this_step
                for param_group in optimizer.param_groups[2:]:
                    param_group['lr'] = args.non_bert_learning_rate
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            slot_predicts = model.predict_slot_filling(
                slot_filling_output, output_mask)
            ent_cls_predicts, ent_cls_top3 = model.predict_ent_classify(
                ent_cls_output)
            rel_cls_predicts, _ = model.predict_rel_classify(
                rel_cls_output, ent_cls_top3)
            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            label_ids = label_ids.cpu()
            ent_ids, rel_ids = ent_ids.cpu(), rel_ids.cpu()

            if len(label_ids) != len(slot_predicts):
                pdb.set_trace()

            train_slot_acc, train_slot_f1, train_ent_cls_acc, train_ent_cls_f1, train_rel_cls_acc, train_rel_cls_f1 = model.acc_f1(
                slot_predicts, label_ids, ent_cls_predicts, ent_ids, rel_cls_predicts, rel_ids)
            pbar.show_process(train_ent_cls_acc, train_loss.item(
            ), train_ent_cls_f1, time.time() - start, step)

            # add tensorboard
            writer.add_scalar('tr_slot_loss_%d' % k, tr_slot_loss.item())
            writer.add_scalar('tr_slot_acc_%d' % k, train_slot_acc)
            writer.add_scalar('tr_slot_f1_%d' % k, train_slot_f1)
            writer.add_scalar('tr_ent_loss_%d' % k, tr_ent_loss.item())
            writer.add_scalar('tr_ent_acc_%d' % k, train_ent_cls_acc)
            writer.add_scalar('tr_ent_f1_%d' % k, train_ent_cls_f1)
            writer.add_scalar('tr_rel_loss_%d' % k, tr_rel_loss.item())
            writer.add_scalar('tr_rel_acc_%d' % k, train_rel_cls_acc)
            writer.add_scalar('tr_rel_f1_%d' % k, train_rel_cls_f1)

        # -----------------------验证----------------------------
        model.eval()
        count = 0
        slot_predicts, ent_predicts, rel_predicts, slot_labels, ent_labels, rel_labels = [
        ], [], [], [], [], []
        ent_predicts_top3, rel_predicts_top3 = [], []
        qids = []
        eval_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                vl_idx, input_ids, input_mask, segment_ids, output_mask, label_ids, ent_ids, rel_ids = batch
                slot_filling_output, ent_cls_output, rel_cls_output = model(
                    input_ids, segment_ids, input_mask, batch_idx=0)
                slot_filling_output, ent_cls_output, rel_cls_output = slot_filling_output.cpu(
                ), ent_cls_output.cpu(), rel_cls_output.cpu()

                eval_slot_loss, eval_ent_loss, eval_rel_loss = model.loss_fn(slot_filling_output, ent_cls_output,
                                                                             rel_cls_output, output_mask, label_ids,
                                                                             ent_ids.cpu(), rel_ids.cpu())
                eval_los = eval_slot_loss + eval_ent_loss + eval_rel_loss
                eval_loss = eval_los + eval_loss
                count += 1
                slot_predict = model.predict_slot_filling(
                    slot_filling_output, output_mask)
                ent_cls_predict, ent_cls_top3 = model.predict_ent_classify(
                    ent_cls_output)
                rel_cls_predict, rel_cls_top3 = model.predict_rel_classify(
                    rel_cls_output, ent_cls_top3)
                slot_predicts.append(slot_predict)
                ent_predicts.append(ent_cls_predict)
                rel_predicts.append(rel_cls_predict)
                ent_predicts_top3.append(ent_cls_top3)
                rel_predicts_top3.append(rel_cls_top3)

                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                slot_labels.append(label_ids)
                ent_labels.append(ent_ids)
                rel_labels.append(rel_ids)
                qids.extend(vl_idx.tolist())

            eval_slot_predicted = torch.cat(slot_predicts, dim=0).cpu()
            eval_slot_labeled = torch.cat(slot_labels, dim=0).cpu()
            eval_ent_predicted = torch.cat(ent_predicts, dim=0).cpu()
            eval_ent_labeled = torch.cat(ent_labels, dim=0).cpu()
            eval_rel_predicted = torch.cat(rel_predicts, dim=0).cpu()
            eval_rel_labeled = torch.cat(rel_labels, dim=0).cpu()
            eval_ent_top3s = torch.cat(ent_predicts_top3, dim=0).cpu()
            eval_rel_top3s = torch.cat(rel_predicts_top3, dim=0).cpu()

            eval_slot_acc, eval_slot_f1, eval_ent_cls_acc, eval_ent_cls_f1, eval_rel_cls_acc, eval_rel_cls_f1 = model.acc_f1(
                eval_slot_predicted, eval_slot_labeled, eval_ent_predicted,
                eval_ent_labeled, eval_rel_predicted, eval_rel_labeled)
            model.class_report(eval_slot_predicted, eval_slot_labeled, eval_ent_predicted,
                               eval_ent_labeled, eval_rel_predicted, eval_rel_labeled, e + 1)

            eval_overall_top1, eval_overall_top3, is_trues_top1, is_true_top3 = model.overall_f1(
                eval_ent_top3s, eval_ent_labeled, eval_rel_top3s, eval_rel_labeled)

            logger.info(
                '\n\nEpoch %d \ntrain_loss: %4f - eval_loss: %4f \ntrain_slot_acc:%4f - eval_slot_acc:%4f - eval_slot_f1:%4f\ntrain_ent_acc: %4f - eval_ent_acc: %4f - eval_ent_f1: %4f\ntrain_rel_acc: %4f - eval_rel_acc: %4f - eval_rel_f1: %4f\noverall_top1_f1: %4f - overall_top3_f1: %4f\n'
                % (e + 1,
                   train_loss.item(),
                   eval_loss.item() / count,
                   train_slot_acc,
                   eval_slot_acc,
                   eval_slot_f1,
                   train_ent_cls_acc,
                   eval_ent_cls_acc,
                   eval_ent_cls_f1,
                   train_rel_cls_acc,
                   eval_rel_cls_acc,
                   eval_rel_cls_f1,
                   eval_overall_top1,
                   eval_overall_top3))
            # add tensorboard
            writer.add_scalar('eval_slot_loss_%d' % k, eval_slot_loss.item())
            writer.add_scalar('eval_slot_acc_%d' % k, eval_slot_acc)
            writer.add_scalar('eval_slot_f1_%d' % k, eval_slot_f1)
            writer.add_scalar('eval_ent_loss_%d' % k, eval_ent_loss.item())
            writer.add_scalar('eval_ent_acc_%d' % k, eval_ent_cls_acc)
            writer.add_scalar('eval_ent_f1_%d' % k, eval_ent_cls_f1)
            writer.add_scalar('eval_rel_loss_%d' % k, eval_rel_loss.item())
            writer.add_scalar('eval_rel_acc_%d' % k, eval_rel_cls_acc)
            writer.add_scalar('eval_rel_f1_%d' % k, eval_rel_cls_f1)
            writer.add_scalar('eval_overall_top1_f1_%d' % k, eval_overall_top1)
            writer.add_scalar('eval_overall_top3_f1_%d' % k, eval_overall_top3)

            if e % verbose == 0:
                train_losses.append(train_loss.item())
                eval_losses.append(eval_loss.item() / count)
                eval_slot_accuracy.append(eval_slot_acc)
                eval_ent_accuracy.append(eval_ent_cls_acc)
                eval_rel_accuracy.append(eval_rel_cls_acc)
                eval_slot_f1_score.append(eval_slot_f1)
                eval_ent_f1_score.append(eval_ent_cls_f1)
                eval_rel_f1_score.append(eval_rel_cls_f1)
                eval_overall_top1_f1.append(eval_overall_top1)
                eval_overall_top3_f1.append(eval_overall_top3)

            # 保存最好的模型
            if eval_overall_top1 > best_f1:
                no_increment = 0
                best_f1 = eval_overall_top1
                save_model(model, args.output_dir)
                write_eval_res(args.eval_res_path, qids, eval_ent_predicted.tolist(
                ), eval_rel_predicted.tolist(), is_trues_top1, is_true_top3, k)
            else:
                no_increment += 1
                if no_increment >= 5:
                    break

    loss_acc_f1_plot(history)

    return min(eval_losses), max(eval_slot_f1_score), max(eval_ent_f1_score), max(eval_rel_f1_score), max(
        eval_slot_accuracy), max(eval_ent_accuracy), max(eval_rel_accuracy), max(eval_overall_top1_f1), max(eval_overall_top3_f1)
