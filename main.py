import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from utils import *
from tqdm import tqdm
import logging
import time
from PIL import Image

def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True

    if not os.path.isdir(os.path.join(args.path, './ckpt')):
        os.mkdir(os.path.join(args.path,'./ckpt'))
    if not os.path.isdir(os.path.join(args.path,'./results')):
        os.mkdir(os.path.join(args.path,'./results'))    
    if not os.path.isdir(os.path.join(args.path, './ckpt', args.name)):
        os.mkdir(os.path.join(args.path, './ckpt', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name)):
        os.mkdir(os.path.join(args.path, './results', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name, "log")):
        os.mkdir(os.path.join(args.path, './results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.path, "results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time())))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
        
    args.logger.info("[{}] starts".format(args.name))
    
    # 1. load data
    
    if args.distilation:
         transform_cifar = transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.RandomResizedCrop(28),
                                  transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform_cifar = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.RandomResizedCrop(28),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
    dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.path, args.data_dir), train=True, download=True, transform=transform_cifar)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    # 2. setup
    
    args.logger.info("setting up...")
    model = torchvision.models.resnet18() if args.distilation else torchvision.models.resnet50()
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

    if args.load:
    	loaded_data = load(args, args.ckpt)
    	model.load_state_dict(loaded_data['model'])
    	optimizer.load_state_dict(loaded_data['optimizer'])

    # 3. train / test
    
    if not args.test:
        # train
        args.logger.info("starting training")
        train_loss_meter = AverageMeter(args, name="Loss", save_all=True, x_label="epoch")
        val_acc_meter = AverageMeter(args, name="Val-Acc", save_all=True, x_label="epoch")
        steps = 1
        for epoch in range(1, 241):
            if steps < args.start_from_step:
                steps += 1
                optimizer.zero_grad()
                optimizer.step()
                scheduler.step()
                continue
                
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter(args)
            for data, answers in tqdm(loader):
                optimizer.zero_grad()
                batch = data.shape[0]
                preds = model(data.to(args.device))
                loss = loss_fn(preds, answers.to(args.device))
                loss.backward()
                optimizer.step()
                
                train_loss_tmp_meter.update(loss, weight=batch)
                steps += 1
            scheduler.step()
            
            # validate and save
            model.eval()
            tmp_val_acc_meter = AverageMeter(args)
            with torch.no_grad():
                for data, answers in tqdm(val_loader):
                    batch = data.shape[0]
                    preds = model(data.to(args.device))
                    corrects, total = val_check(preds, answes.to(args.device))
                    tmp_val_acc_meter.update((corrects / total) * 100, weight=total)
            val_acc = tmp_val_acc_meter.avg
            val_acc_meter.update(val_acc)
            
            if steps % args.save_period == 0:
                save(args, "epoch_{}".formate(epoch), {'model': model.state_dict()})
                
                args.logger.info("[{}] plot recorded".format(steps, spent_time))
                train_loss_meter.save()
                val_acc_meter.save()
                args.logger.infor("[{}] plot saved".format(epoch))
                
            spent_time = time.time() - spent_time
            args.logger.info("[{}] train loss: {:.3f} validation accuracy: {:.3f} took {:.1f} seconds".format(epoch, train_loss_meter.val, val_acc_meter.val, spent_time))

    else:
        # test
        args.logger.info("starting test")
        spent_time = time.time()
        model.eval()
        test_acc_meter = AverageMeter(args, name="Test-Acc", save_all=True, x_label="epoch")
        tmp_test_acc_meter = AvverageMeter(args)
        with torch.no_grad():
            for data, answers in tqdm(val_loader):
                batch = data.shape[0]
                preds = model(data.to(args.device))
                corrects, total = val_check(preds, answes.to(args.device))
                tmp_test_acc_meter.update((corrects / total) * 100, weight=total)
        spent_time = time.time() - spent_time
        test_acc = tmp_test_acc_meter.avg
        test_acc_meter.update(test_acc)
        args.logger.info("[{}] test accuracy: {:.3f} took {:.1f} seconds".format(epoch, test_acc_meter.val, spent_time))
        test_acc_meter.save()

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='CRD')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--path',
        type=str,
        default='.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5)
    parser.add_argument(
    	'--warmup',
    	type=int,
    	default=5),
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=5)
    parser.add_argument(
        '--start_from_step',
        type=int,
        default=1)
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='_')
    parser.add_argument(
        '--load',
        action='store_true')
    
    
    
    parser.add_argument(
        '--distilation',
        action='store_true')

    
    args = parser.parse_args()

        
    main(args)