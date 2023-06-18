import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from datasets import Proj3_Dataset
from models import *


np.random.seed(991108)

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def get_args_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lr', default='1e-4')
    argparser.add_argument('--optim_type', default='adamw')
    argparser.add_argument('--arch_ver', default='ver1')
    argparser.add_argument('--labelsmooth', default='0.1')
    argparser.add_argument('--freeze', action='store_true')
    argparser.add_argument('--weight_decay', default='1e-4')
    argparser.add_argument('--ver_name', default="")
    argparser.add_argument('--epoch', default="100")
    argparser.add_argument('--pretrained', default="")
    argparser.add_argument('--batch', default="256")

    args = argparser.parse_args()

    return args

def split_trainval(num_train=45, num_val=10):
    trainval_annos = pd.read_csv('datasets/train_anno.csv')

    categories = sorted(trainval_annos['cls'].unique())
    train_annos, val_annos = [], []
    for c in categories:
        idxs = np.arange(num_train + num_val)
        np.random.shuffle(idxs)
        tgt_df = trainval_annos.groupby('cls').get_group(c).reset_index(drop=True)
        train_annos.append(tgt_df.loc[idxs[:num_train]])
        val_annos.append(tgt_df.loc[idxs[num_train:]])

    train_annos = pd.concat(train_annos).reset_index(drop=True)
    val_annos = pd.concat(val_annos).reset_index(drop=True)

    return train_annos, val_annos

def run_val_epoch(net, data_loader):
    net.eval()

    sum_loss = 0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for idx, (img, gt_y) in enumerate(data_loader):
            img, gt_y = img.to(device), gt_y.to(device)
            img = (img - img_mean) / img_std

            num_batch = gt_y.shape[0]
            img = img.to(device).float()
            gt_y = gt_y.to(device).long()
            out = net(img)
            _, pred = torch.max(out, 1)
            correct += pred.eq(gt_y.data).sum().item()

            loss = criterion(out, gt_y)
            sum_loss += num_batch*(loss.item())
            num_samples += num_batch

    loss = sum_loss / num_samples
    acc = 100 * correct / num_samples

    return loss, acc

def run_trainval():
    ep = -1
    val_loss, val_acc = run_val_epoch(net, val_loader)
    print(f"[val-{ep + 1}/{num_epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
    writer.add_scalar('ep_loss/val', val_loss, ep+1)
    writer.add_scalar('ep_acc/val', val_acc, ep+1)

    for ep in range(num_epochs):
        net.train()
        
        ep_loss = 0
        ep_pred_y, ep_gt_y = [], []
        start_time = datetime.now()
        writer.add_scalar("lr", optim.param_groups[0]["lr"], ep+1)
        for idx, (img, gt_y) in enumerate(tqdm(train_loader)):
            img, gt_y = img.to(device), gt_y.to(device)
            img = (img - img_mean) / img_std
            pred_y = net(img)
            loss = criterion(pred_y, gt_y)
            optim.zero_grad()
            
            loss.backward()
            # if args.pretrained != "":
            #     nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            
            
            ep_loss += len(gt_y) * loss.item()

            ep_pred_y.append(pred_y.detach().max(dim=1)[1].cpu())
            ep_gt_y.append(gt_y.cpu())

            if (ep+1) % save_intv == 0:
                torch.save(net.state_dict(), osp.join(ckpt_dir, f'ep{ep+1}.pt'))
        scheduler.step()
        
        end_time = datetime.now()
        print(f"Time elapsed {end_time - start_time}")

        ep_pred_y = torch.cat(ep_pred_y)
        ep_gt_y = torch.cat(ep_gt_y)
        train_loss = ep_loss / len(ep_gt_y)
        train_acc = 100 * (ep_gt_y == ep_pred_y).to(float).mean().item()
        val_loss, val_acc = run_val_epoch(net, val_loader)

        print(f"[train-{ep + 1}/{num_epochs}] loss: {train_loss:.6f} | acc: {train_acc:.3f}%")
        print(f"[val-{ep + 1}/{num_epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
        writer.add_scalar('ep_loss/train', train_loss, ep+1)
        writer.add_scalar('ep_loss/val', val_loss, ep+1)
        writer.add_scalar('ep_acc/train', train_acc, ep+1)
        writer.add_scalar('ep_acc/val', val_acc, ep+1)

if __name__ == '__main__':
    args = get_args_parser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = int(args.epoch)
    save_intv = 2
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)

    num_workers = 2
    batch_size = int(args.batch)
    freeze_backbone = args.freeze
    num_cls = 50
    optim_type = args.optim_type
    arch_ver = args.arch_ver
    output_dir = f'outputs/arch{arch_ver}_lr{lr}_freeze{"T" if freeze_backbone else "F"}_optim{optim_type}'
    if args.ver_name != "":
        output_dir += f"_V{args.ver_name}"
    ckpt_dir = osp.join(output_dir, 'ckpt')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    optim_choices = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
    model_choices = {'ver1': R34_ver1, 'attnPool': R34_attnPool, 'convnext': R34_extension}

    ## split train/val datasets randomly - you can modify this randomness
    train_annos, val_annos = split_trainval(num_train=45, num_val=10)

    ## data transform
    ## For inference, you may use 5-crop (4 corners and center) - T.FiveCrop(img_size)
    img_size = 256
    crop_size = 224
    max_rotation = 30
    train_transform = T.Compose(
            [T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(max_rotation),
            T.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            # T.ColorJitter(brightness=0.5, contrast=0.5),
            T.ToTensor(),
            T.RandomErasing(p=0.5),
    ])
    val_transform = T.Compose([T.Resize(img_size), T.CenterCrop(crop_size), T.ToTensor()])

    ## build dataloader
    train_dataset = Proj3_Dataset(train_annos, 'train', train_transform)
    val_dataset = Proj3_Dataset(val_annos, 'val', val_transform)

    print("Train dataset: #", len(train_dataset))
    print("Val dataset: #", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    ## build model
    net = model_choices[arch_ver](num_cls=num_cls, freeze_backbone=freeze_backbone).to(device)

    # pretrained 
    if args.pretrained != "":
        print(f"Loading a pretrained model from {args.pretrained}...")
        net.load_state_dict(torch.load(args.pretrained, map_location=device))
        # unfreeze!
        for param in net.backbone.parameters():
            param.requires_grad = True
        # unfreeze the last layer of backbone
        # for param in net.backbone.parameters():
        #     param.requires_grad = True
        
        # # unfreeze the second-last layer of backbone
        # for param in net.backbone.layer3.parameters():
        #     param.requires_grad = True
        
        # # unfreeze the third-last layer of backbone
        # for param in net.backbone.layer2.parameters():
        #     param.requires_grad = True

    ## train & validation
    img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.labelsmooth)) ## loss function - you can define others
    train_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optim = optim_choices[optim_type](train_parameters, lr=lr/10, weight_decay=weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=lr/100)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, lr*2, steps_per_epoch=len(train_loader), epochs=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
    scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=20, T_mult=2, eta_max=lr*1.3,  T_up=5, gamma=0.5)
    print(f"Train loader size: {len(train_loader)}")
    run_trainval()

