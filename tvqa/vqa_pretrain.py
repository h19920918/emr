import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model.tvqa_abc import ABC
from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from config import BaseOptions
from util import set_seed

from datetime import datetime
import time


def pretrain(cfg):
    set_seed(cfg.seed)
    writer = SummaryWriter(cfg.log_dir)
    cfg.writer = writer

    dset = TVQADataset(cfg)
    # dset.load_vid()
    cfg.vocab_size = len(dset.word2idx)
    model = ABC(cfg)
    if not cfg.no_glove:
        model.load_embedding(dset.vocab_embedding)

    model.cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.lr, weight_decay=cfg.wd)

    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    for epoch in range(cfg.n_epoch):
        if not early_stopping_flag:
            cur_acc = train(cfg, dset, model, criterion, optimizer, epoch, best_acc)

            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= 3:
                    early_stopping_flag = True
        else:
            print("early stopping with valid acc %.4f" % best_acc)
            cfg.writer.export_scalars_to_json(os.path.join(cfg.log_dir, "all_scalars.json"))
            cfg.writer.close()
            break


def train(opt, dset, model, criterion, optimizer, epoch, previous_best_acc):
    dset.set_mode("train")
    model.train()
    torch.set_grad_enabled(True)
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True, collate_fn=pad_collate)

    train_loss = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        model_inputs, targets, _ = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                     device=opt.device)
        outputs = model(*model_inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        train_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        train_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()
        if batch_idx % opt.log_freq == 1:
            niter = epoch * len(train_loader) + batch_idx

            train_acc = sum(train_corrects) / float(len(train_corrects))
            train_loss = sum(train_loss) / float(len(train_corrects))
            opt.writer.add_scalar("Train/Acc", train_acc, niter)
            opt.writer.add_scalar("Train/Loss", train_loss, niter)

            # Test
            valid_acc, valid_loss = validate(opt, dset, model, criterion, mode="valid")
            opt.writer.add_scalar("Valid/Loss", valid_loss, niter)

            valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            valid_acc_log.append(valid_log_str)
            if valid_acc > previous_best_acc:
                previous_best_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(opt.log_dir, "best_valid.pth"))
            print(" Train Epoch %d loss %.4f acc %.4f Val loss %.4f acc %.4f"
                  % (epoch, train_loss, train_acc, valid_loss, valid_acc))

            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []

        if opt.debug:
            break

    # additional log
    with open(os.path.join(opt.log_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc

def validate(opt, dset, model, criterion, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    valid_qids = []
    valid_loss = []
    valid_corrects = []
    for _, batch in enumerate(valid_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        outputs = model(*model_inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if opt.debug:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    return valid_acc, valid_loss
