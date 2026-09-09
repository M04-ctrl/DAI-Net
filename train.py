# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import MultiBoxLoss, EnhanceLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from PIL import Image

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='dark', type=str,
                    choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local-rank', '--local_rank',
                    type=int,
                    help='local rank for dist')

args = parser.parse_args()

import torch.distributed as dist

# 1. 统一获取 local_rank
local_rank = int(os.environ.get('LOCAL_RANK', 0))

if torch.cuda.is_available():
    if args.cuda:
        # 2. 初始化分布式进程组（仅一次）
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        # 3. 设置当前进程使用的 GPU
        torch.cuda.set_device(local_rank)
        torch.set_default_dtype(torch.float32)
        # [FIX 5] 删除 torch.set_default_device：该调用会全局劫持张量创建设备，
        #          与 DataLoader worker 子进程冲突，导致多卡训练挂起或报错。

        if local_rank == 0:
            print(f'Successfully initialized! Using {torch.cuda.device_count()} GPUs.')
    else:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float32)

# 统一文件夹创建
save_folder = os.path.join(args.save_folder, args.model)
# 只让主进程创建文件夹，避免多进程竞争
if local_rank == 0 and not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=detection_collate,
                               sampler=train_sampler,
                               pin_memory=True)
val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=0,
                             collate_fn=detection_collate,
                             sampler=val_sampler,
                             pin_memory=True)

min_loss = np.inf


def train():
    per_epoch_size = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    start_epoch = 0
    iteration = 0
    step_index = 0

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()
    # [FIX 7] 权重文件保存时 tensor 带有 CUDA 设备元信息，多进程并发加载时
    #          旧版 pickle loader 无法用 map_location='cuda:N' 正确覆盖，
    #          导致 "storage on different device" 报错。
    #          标准做法：统一先加载到 CPU，再由后续 .cuda() 搬运到各自 GPU。
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth',
                                       map_location='cpu'))

    # ckpt_data 暂存，optimizer 创建后再恢复其 state
    ckpt_data = None
    if args.resume:
        if local_rank == 0:
            print('Resuming training, loading {}...'.format(args.resume))
        # [FIX 8] 完整恢复 checkpoint：模型权重、epoch、iteration、step_index
        #          resume 应指向 dsfd_checkpoint.pth（含上述字段）
        ckpt_data = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt_data['weight'])
        start_epoch = ckpt_data['epoch'] + 1          # 从下一个 epoch 继续
        iteration   = ckpt_data.get('iteration', start_epoch * per_epoch_size)
        step_index  = ckpt_data.get('step_index', 0)
    else:
        base_weights = torch.load(args.save_folder + basenet,
                                  map_location='cpu')
        if local_rank == 0:
            print('Load base network {}'.format(args.save_folder + basenet))
        if args.model == 'vgg' or args.model == 'dark':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)

    if not args.resume:
        if local_rank == 0:
            print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()), 4)
    param_group = []
    param_group += [{'params': dsfd_net.vgg.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.ref.parameters(), 'lr': lr / 10.}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # [FIX 8 cont.] optimizer 构建完毕后立即恢复 state，并补偿历史 LR 衰减
    if ckpt_data is not None and 'optimizer' in ckpt_data:
        optimizer.load_state_dict(ckpt_data['optimizer'])
        if local_rank == 0:
            print(f'Optimizer state restored. lr = {optimizer.param_groups[0]["lr"]:.2e}')
    elif ckpt_data is not None:
        # 旧格式 checkpoint 无 optimizer 字段：手动补偿 LR 衰减
        for s in cfg.LR_STEPS:
            if iteration > s:
                for pg in optimizer.param_groups:
                    pg['lr'] *= args.gamma
        if local_rank == 0:
            print(f'[Warning] No optimizer state in checkpoint. '
                  f'LR manually decayed to {optimizer.param_groups[0]["lr"]:.2e}')

    if args.cuda:
        if args.multigpu:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(), find_unused_parameters=True)
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
        # [FIX 3] 修正拼写错误：benckmark → benchmark
        cudnn.benchmark = True
        # [FIX 9] optimizer state 用 map_location='cpu' 加载后，momentum 缓冲区在 CPU。
        #          必须在 net.cuda() / DDP 包装完成之后，把所有 state tensor 搬到对应 GPU，
        #          否则 optimizer.step() 时参数(CUDA) vs 缓冲区(CPU) 设备不一致而报错。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    criterion = MultiBoxLoss(cfg, args.cuda)
    criterion_enhance = EnhanceLoss()
    if local_rank == 0:
        print('Loading wider dataset...')
        print('Using the specified args:')
        print(args)

    # LR 衰减补偿已在 optimizer 恢复块中处理，此处无需重复
    net_enh.eval()
    net.train()

    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0

        # [FIX 4] 必须在每个 epoch 开始前调用 set_epoch，
        #          否则所有 epoch 的 shuffle 顺序完全相同
        train_sampler.set_epoch(epoch)

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = Variable(images.cuda() / 255.)
            targetss = [Variable(ann.cuda(), requires_grad=False)
                        for ann in targets]
            img_dark = torch.empty_like(images)
            for i in range(images.shape[0]):
                img_dark[i], _ = Low_Illumination_Degrading(images[i])

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            R_dark_gt, I_dark = net_enh(img_dark)
            R_light_gt, I_light = net_enh(images)

            out, out2, loss_mutual = net(img_dark, images,
                                        I_dark.detach(), I_light.detach())
            R_dark, R_light, R_dark_2, R_light_2 = out2

            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss_enhance = criterion_enhance(
                [R_dark, R_light, R_dark_2, R_light_2,
                 I_dark.detach(), I_light.detach()],
                images, img_dark) * 0.1
            loss_enhance2 = (F.l1_loss(R_dark, R_dark_gt.detach()) +
                             F.l1_loss(R_light, R_light_gt.detach()) +
                             (1. - ssim(R_dark, R_dark_gt.detach())) +
                             (1. - ssim(R_light, R_light_gt.detach())))

            loss = (loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2 +
                    loss_enhance2 + loss_enhance + loss_mutual)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                if local_rank == 0:
                    print('Timer: %.4f' % (t1 - t0))
                    print('epoch:' + repr(epoch) + ' || iter:' +
                          repr(iteration) + ' || Loss:%.4f' % (tloss))
                    print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                        loss_c_pal1.item(), loss_l_pa1l.item()))
                    print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                        loss_c_pal2.item(), loss_l_pa12.item()))
                    print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                if local_rank == 0:
                    print('Saving state, iter:', iteration)
                    file = 'dsfd_' + repr(iteration) + '.pth'
                    torch.save(dsfd_net.state_dict(),
                               os.path.join(save_folder, file))
            iteration += 1

        if (epoch + 1) >= 0:
            val(epoch, net, dsfd_net, net_enh, criterion, optimizer, iteration, step_index)
        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, net_enh, criterion, optimizer, iteration, step_index):
    net.eval()
    step = 0
    losses = torch.tensor(0.).cuda()
    t1 = time.time()

    # [FIX 1] volatile=True 在 PyTorch ≥ 0.4 中已废弃并会报错，
    #          改为 torch.no_grad() 上下文管理器
    with torch.no_grad():
        for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
            if args.cuda:
                images = images.cuda() / 255.
                targets = [ann.cuda() for ann in targets]
            else:
                images = images / 255.

            img_dark = torch.stack(
                [Low_Illumination_Degrading(images[i])[0]
                 for i in range(images.shape[0])],
                dim=0)
            out, R = net.module.test_forward(img_dark)

            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
            loss = loss_l_pa12 + loss_c_pal2

            # [FIX 2] 保持为 CUDA tensor（不调用 .item()），
            #          dist.reduce 要求操作数必须是 tensor
            losses += loss
            step += 1

    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)

    # [FIX 6] 转为 Python 标量再与 np.inf 比较，避免 tensor 比较的隐患
    tloss = (losses / step / torch.cuda.device_count()).item()
    t2 = time.time()
    if local_rank == 0:
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        if local_rank == 0:
            print('Saving best state, epoch', epoch)
            torch.save(dsfd_net.state_dict(),
                       os.path.join(save_folder, 'dsfd.pth'))
        min_loss = tloss

    # [FIX 8 cont.] 保存完整 checkpoint：epoch/iteration/step_index/模型/optimizer
    #                 下次 --resume 时可精确恢复所有训练状态
    states = {
        'epoch':      epoch,
        'iteration':  iteration,
        'step_index': step_index,
        'weight':     dsfd_net.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }
    if local_rank == 0:
        torch.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by gamma at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma


if __name__ == '__main__':
    train()