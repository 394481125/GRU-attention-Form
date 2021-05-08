from Dataset import VideoDataset
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from Models import Spatial_TemporalNet
import torch.backends.cudnn as cudnn
import time
import os
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import scipy.io as scio
import math
import torch.nn.functional as F
import argparse

# 时空连续性

def get_args_parser():
    parser = argparse.ArgumentParser('Global and Local Knowledge-Aware Attention Network for Action Recognition')
    # ============================ Code Configs ============================
    parser.add_argument('--train_video_list', default='list/hmdb_split1_train.txt',
                        type=str)  # ucf_trans_split1_train.txt
    parser.add_argument('--test_video_list', default='list/hmdb_split1_test.txt', type=str)
    parser.add_argument('--root', default='/content/GRU-attention-Form/', type=str)
    parser.add_argument('--dataset', default='hmdb', type=str)
    parser.add_argument('--target_dataset', default='hmdb', type=str)
    parser.add_argument('--model_dir', default='/content/drive/MyDrive/GRU-attention-Form/model/', type=str)
    parser.add_argument('--get_scores', default=False, type=bool)
    parser.add_argument('--description', type=str)
    parser.add_argument('--pretrained', default=False, type=bool)
    parser.add_argument('--cross', default='True', type=str)

    # ============================ Learning Configs =============test_loader===============
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=str)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--lr_step', default=[30, 40], type=list)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)

    # ============================ Model Configs ============================
    parser.add_argument('--attention_type', default='all', type=str, help='average, auto and learned, all')
    parser.add_argument('--feature_size', default=512, type=int)
    parser.add_argument('--hidden_size', default=1024, type=int)
    parser.add_argument('--segments', default=12, type=int)
    parser.add_argument('--frames', default=1, type=int)
    parser.add_argument('--base_model', default='resnet34', type=str)
    parser.add_argument('--kernel_size', default=7, type=int)

    return parser

def main(args):

    global best_prec1
    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
    cudnn.benchmark = True
    # 如果想要避免这种结果波动，设置：
    cudnn.deterministic = True


    # 创建模型文件夹
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    strat_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 训练数据加载
    train_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.train_video_list, num_segments=args.segments,
                     num_frames=args.frames,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.RandomCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # 加载测试数据
    test_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.test_video_list, num_segments=args.segments,
                     num_frames=args.frames, test_mode=True,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # 网络模型创建
    net = Spatial_TemporalNet(basemodel=args.base_model, dataset=args.dataset, segment=args.segments, attention_type=args.attention_type,
                                      hidden_size=args.hidden_size, img_dim=args.feature_size, kernel_size=args.kernel_size)
    # 多GPU训练的代码
    net = torch.nn.DataParallel(net).cuda()

    # 获得梯度为True
    for param in net.parameters():
        param.requires_grad = True

    # 注意力损失函数
    def attentionloss(baseline, attention, target):
        target_temp = target.unsqueeze(1)
        baseline_temp = torch.gather(baseline, 1, target_temp).squeeze(0)
        attention_temp = torch.gather(attention, 1, target_temp).squeeze(0)
        selfloss = torch.max(torch.zeros(1).cuda(), baseline_temp - attention_temp + 0.1)

        return selfloss.mean()

    #
    # if args.cross:
    #     net.load_state_dict(torch.load('./model/model.pkl'))
    #     if args.target_dataset == 'hmdb':
    #         num_class = 51
    #     if args.target_dataset == 'ucf':
    #         num_class = 101
    #     setattr(net.module.temporal.reason_learned, 'fc', nn.Linear(1024, num_class).cuda())
    #     setattr(net.module.temporal.reason_auto, 'fc', nn.Linear(1024, num_class).cuda())
    #     setattr(net.module.temporal.reason_average, 'fc', nn.Linear(1024, num_class).cuda())
    #     print ('load pre-trained weights on Kinetics successfully ')
    #
    # if args.get_scores:
    #     net.load_state_dict(torch.load('./model/model.pkl'))
    #     print ('begin to get class scores model')

    # 分类损失为交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    print ('doing experiments on ' + args.test_video_list)
    best_prec1 = 0
    # 迭代训练并优化
    for epoch in range(args.epoch):
        if not args.get_scores:
            adjust_learning_rate(optimizer, epoch, args)
            # 训练
            train_model(train_loader, net, criterion, optimizer, epoch, attentionloss)

            if (epoch + 1) % args.eval_freq == 0:
                # 测试
                prec1 = test_model(test_loader, net, epoch)

                if prec1 > best_prec1:
                    best_prec1 = prec1
                    torch.save(net.state_dict(), os.path.join(args.model_dir, strat_time + '.pkl'))
                    print('保存在',os.path.join(args.model_dir, strat_time + '.pkl'))
        else:
            print ('开始获得分数')
            gets(test_loader, net, epoch, args)
            print ('完成')
            break

def train_model(train_loader, net, criterion, optimizer, epoch, attentionloss):
    net.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()
    for step, (input, target) in enumerate(train_loader):
        # 输入矩阵
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        # 转化为GPU版
        target = Variable(target).cuda()
        # 传入网络获得输出
        output_average, output_auto, output_learned, output = net(input)
        # 对输出进行softmax
        output_average = F.softmax(output_average, dim=1)
        output_auto = F.softmax(output_auto, dim=1)
        output_learned = F.softmax(output_learned, dim=1)

        # 计算综合损失函数
        loss = criterion(output, target) + attentionloss(output_average, output_auto, target) + attentionloss(output_average, output_learned, target)

        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        losses.update(loss.item(), input.size(0))

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器优化
        optimizer.step()
        # 更新时间
        batch_time.update(time.time() -start)
        start = time.time()

        if (step + 1) % args.print_freq == 0:
            NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            output = ('Now Time {0}  Epoch:{1} || Step:{2}'
                      ' || Loss:{loss.avg:.4f}'
                      ' || Time:{batch_time.avg:.3f}'.format(NowTime, epoch, step + 1, loss=losses, batch_time=batch_time))
            print (output)

    accuracy = ('Epoch:{0} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(epoch + 1, top1=top1, top5=top5)
    print (accuracy)

def test_model(test_loader, net, epoch):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for input, target in tqdm(test_loader):
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        target = Variable(target).cuda()

        output_average, output_auto, output_learned, output = net(input)
        output = torch.mean(output, dim=0, keepdim=True)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1)
        top5.update(prec5)

    NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = ('Testing Phrase ==>> Now Time {0} Epoch:{1} || Best Accuracy:{2} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(NowTime, epoch, max(best_prec1, top1.avg), top1=top1, top5=top5, )
    print (accuracy)


    return top1.avg

def gets(test_loader, net, epoch, args):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.target_dataset == 'hmdb':
        mat = np.zeros((1, 51))
    elif args.target_dataset == 'ucf':
        mat = np.zeros((1, 101))
    elif args.target_dataset == 'kinetic':
        mat = np.zeros((1, 600))
    for step, (input, target) in enumerate(test_loader):
        print ('The Testing Number is {0}'.format(step))
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        target = Variable(target).cuda()

        output_average, output_auto, output_learned, output = net(input)
        output = torch.mean(output, dim=0, keepdim=True)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        mat = np.vstack((mat, output.cpu().data.view(1, -1).numpy()))

        top1.update(prec1)
        top5.update(prec5)


    NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = ('Testing Phrase ==>> Now Time {0} Epoch:{1} || Best Accuracy:{2} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(NowTime, epoch, max(best_prec1, top1.avg), top1=top1, top5=top5, )
    print (accuracy)
    df = pd.DataFrame(mat[1:])
    df.to_excel(args.target_dataset + '.xlsx')

def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    corrrect = pred.eq(target.view(-1, 1).expand_as(pred))

    store = []
    for k in topk:
        corrrect_k = corrrect[:,:k].float().sum()
        store.append(corrrect_k * 100.0 / batch_size)
    return store

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.lr_step:
        args.learning_rate = args.learning_rate * 0.2
    lr = 0.5 * (1 + math.cos(epoch * math.pi / args.epoch)) * args.learning_rate

    # lr = lr * 0.1 ** (epoch // lr_step)
    print ('the learning rate is changed to {0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)