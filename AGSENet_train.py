import argparse
import os
from collections import OrderedDict
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as albu
import random
import AGSENet_6
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
ARCH_NAMES =AGSENet_6.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')




"""

指定参数：
--dataset dsb2018_96 
--arch NestedUNet

"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='AGSENet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: AGSENet)')
    #parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--in_ch', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='Puddle-1000 Dataset',
                        help='dataset name')
    parser.add_argument('--dataset_train', default='Puddle-1000 Dataset_train_night',
                        help='dataset name')
    parser.add_argument('--dataset_val', default='Puddle-1000 Dataset_val_night',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

def set_rand_seed(seed=42):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(config, train_loader, model, criterion,optimizer, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'loss0': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        if epoch >= 50:
            model.sse0.requires_grad_(True)
            model.sse1.requires_grad_(True)
            model.sse2.requires_grad_(True)
            model.sse3.requires_grad_(True)
            model.sse4.requires_grad_(True)
        else:
            model.sse0.requires_grad_(False)
            model.sse1.requires_grad_(False)
            model.sse2.requires_grad_(False)
            model.sse3.requires_grad_(False)
            model.sse4.requires_grad_(False)

        # compute output
        output, hx1, hx2, hx3, hx4, hx5, hx6 = model(input)
        loss0 = criterion(output, target)
        loss1 = criterion(hx1, target)
        loss2 = criterion(hx2, target)
        loss3 = criterion(hx3, target)
        loss4 = criterion(hx4, target)
        loss5 = criterion(hx5, target)
        loss6 = criterion(hx6, target)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        iou = iou_score(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['loss0'].update(loss0.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))


        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('loss0', avg_meters['loss0'].avg),
            ('iou', avg_meters['iou'].avg)

        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('loss0', avg_meters['loss0'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'loss0': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output, hx1, hx2, hx3, hx4, hx5, hx6 = model(input)
            loss0 = criterion(output, target)
            loss1 = criterion(hx1, target)
            loss2 = criterion(hx2, target)
            loss3 = criterion(hx3, target)
            loss4 = criterion(hx4, target)
            loss5 = criterion(hx5, target)
            loss6 = criterion(hx6, target)
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['loss0'].update(loss0.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('loss0', avg_meters['loss0'].avg),
                ('iou', avg_meters['iou'].avg)

            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('loss0', avg_meters['loss0'].avg),
                        ('iou', avg_meters['iou'].avg)])




def main():
    set_rand_seed(42)
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s_1030agsenet_night_256' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = AGSENet_6.__dict__[config['arch']](config['num_classes'], config['in_ch'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=8, T_mult=2, eta_min=config['min_lr'])
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids_train = glob(os.path.join('inputs', config['dataset_train'], 'images', '*' + config['img_ext']))
    img_ids_train = [os.path.splitext(os.path.basename(p))[0] for p in img_ids_train]
    train_img_ids = img_ids_train

    img_ids_val = glob(os.path.join('inputs', config['dataset_val'], 'images', '*' + config['img_ext']))
    img_ids_val = [os.path.splitext(os.path.basename(p))[0] for p in img_ids_val]
    val_img_ids = img_ids_val

    # 数据增强：
    train_transform = Compose([
        albu.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast()
        ], p=1),  # 按照归一化的概率选择执行哪一个
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset_train'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset_train'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset_val'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset_val'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('loss0', []),
        ('iou', []),
        ('val_loss', []),
        ('val_loss0', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step()
        elif config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        print('loss %.4f - loss0 %.4f - iou %.4f - val_loss %.4f - val_loss0 %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['loss0'], train_log['iou'],val_log['loss'],val_log['loss0'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['loss0'].append(train_log['loss0'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_loss0'].append(val_log['loss0'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
