from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import  transforms
import pickle
import os
import os.path
import datetime
import numpy as np
from data.highDSSl_loader_barlow_twins_correctedArgo import MyDataset,DataLoader
from utils.util import AverageMeter, cluster_acc, barlow_twin_loss,LARS,GaussianBlur,Solarization,plot_umap,debug_cluster,barlow_twin_loss_v2
from tqdm import tqdm
import shutil
from models.resnet3d_self_supervised import ResNet, BasicBlock, generate_model 
from models.timesformer_bw import TimeSformer
import math
import matplotlib.pyplot as plt
from einops import rearrange
from tensorboardX import SummaryWriter
import json
from utils.cosine_annealing import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import LambdaLR
import random
import torchio
from PIL import Image, ImageOps, ImageFilter
import pytorch_warmup as warmup
from pytorchvideo.transforms import (
    RandomTemporalSubsample,
    RandomResizedCrop,
    UniformTemporalSubsample,
    )
import torch_optimizer as optim

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        #lr_schedule = np.linspace(final_lr, warmup_lr, decay_iter)
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
            
def visualize_scenarios(data,flag,args,sample):
    for k in range(2):
        for i in range(args.img_depth):
            if flag==1:
                a = data[k,i,:,:,:]
            else:
                a = data[k,:,i,:,:]
            if sample==1:
                name = 'normal'
            else:
                name = 'augumented'
            aa = a.reshape(args.img_height,args.img_width)
            stdName = args.save_dir+'\\'+name+'_OG_'+str(k)+'_' +str(i)+'.png'
            plt.imshow(aa)
            plt.savefig(stdName)

def save_checkpoint(model, optimizer, save_path, epoch, lr_scheduler):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_dict': lr_scheduler.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path,lr_scheduler):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_dict'])  
    return model, optimizer, lr_scheduler, epoch

def train(epoch, model, device, dataloader, optimizer,args):
    loss_record = AverageMeter()
    model.train()
    batch_pass = 0
    for batch_idx, ((x1,x2),y,idx) in enumerate(tqdm(dataloader(epoch,batch_pass)),start=epoch * len(dataloader(epoch,batch_pass))):
        # shape of data (batch x channel x frame x height x width)
 
        batch_pass = batch_pass+1
        if  args.model_use == 'Times':
            x1 = rearrange(x1, 'b c f h w  -> b f c h w ')
            x2 = rearrange(x2, 'b c f h w  -> b f c h w ')

        if args.viz == True and args.model_use == 'Times':
            visualize_scenarios(x1,1,args,1)
            visualize_scenarios(x2,1,args,2)

        elif args.viz==True:
            visualize_scenarios(x1,0,args,1)
            visualize_scenarios(x2,0,args,2)
        x1 = x1.float()
        x2 = x2.float()           
        x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True) # add this line
        output1,feature1 = model(x1)
        output2,feature2 = model(x2)
        check = int((output1 != output1).sum())
        if(check>0):
            print("your data contains Nan")
            print(output1)

        loss = barlow_twin_loss(output1, output2,device,args.save_dir,args.lambda_param,visualize=True,epoch=batch_idx,args=args)
        loss_record.update(loss.item(), x1.size(0))
        if batch_idx%50==0:
            print('Train Epoch: {} Step {}  Avg Loss: {:.4f} '.format(epoch, batch_idx,  loss))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        args.writer.add_scalar('learning rate',get_lr(optimizer),batch_idx)

    print('--------------------------------------------------------------')
    print('--------------------------------------------------------------')

    print('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

    return loss_record

def test(model, device, dataloader, epoch, args):
    model.eval()
    num_unlab = len(dataloader(0).dataset)
    output =  torch.zeros((num_unlab,512))
    output = output.to(device)
    targets = torch.zeros((num_unlab,),dtype=int)
    targets = targets.to(device)
    total_num = 0
    total_correct_1 = 0.0
    for batch_idx, ((x1,x2),y,idx) in enumerate(tqdm(dataloader())):
        if args.viz==True and args.model_use == 'Times':
            visualize_scenarios(x1,1,args)
        elif args.viz==True:
            visualize_scenarios(x1,0,args,1)
            visualize_scenarios(x2,0,args,2)

        if  args.model_use == 'Times':
            x1 = rearrange(x1, 'b c f h w  -> b f c h w ')
            x2 = rearrange(x2, 'b c f h w  -> b f c h w ')
        x1 = x1.float()
        x2 = x2.float()
        x1, x2, y= x1.to(device, non_blocking=True), x2.to(device, non_blocking=True), y.to(device, non_blocking=True) # add this line
        with torch.no_grad():
            output1,feature1 = model(x1)
        output[idx,:] = feature1
        targets[idx,] = y.long()
        total_num+=x1.shape[0]
    # cluster_acc(targets, classChec)
    acc = debug_cluster(output.cpu().numpy(),targets.cpu().numpy(),num_unlabeled_classes=args.classes)
    #plot_umap(output, targets, epoch,args.classes, args)    
    print('Test Acc: {:.4f}'.format(acc))
    return acc 
def exclude_bias_and_norm(p):
    return p.ndim == 1

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    # Training settings
    parser = argparse.ArgumentParser(description='Rot_resNet')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                                    help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.0003, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='ssl_net_120')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--og_choose', type=list, default=[0,3,6,9])
    parser.add_argument('--model_use', type=str, default='Times') #Times
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--img_width', type=int, default=120)
    parser.add_argument('--img_height', type=int, default=120)
    parser.add_argument('--img_depth', type=int, default=4)
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--lambda_param', type=float, default=0.0051)
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
  
    parser.add_argument("--warmup_lr", type=float, default=0, help="warmup learning rate")
    parser.add_argument("--base_lr", type=float, default=0.02, help="base learning rate")

    args = parser.parse_args()
  
    if args.model_use == 'ResNet':
        args.model_name = args.model_name + '_resnet'
    else:
        args.model_name = args.model_name + '_times'

    devices = torch.cuda.device_count()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_class = math.factorial(len(args.og_choose))
    torch.manual_seed(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name,args.model_use,str(args.model_depth))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 
    args.save_dir = model_dir
    summar_dir = args.save_dir+'/runs/'+ str(args.model_name)
    if not os.path.exists(summar_dir):
        os.makedirs(summar_dir)

    writer = SummaryWriter(summar_dir)
    # train_transforms = transforms.Compose([
    #     RandomTemporalSubsample(4),
    #     #transforms.Resize(80),
    #     transforms.RandomCrop(80),
    #     transforms.RandomRotation(degrees=(-10, 10), fill=(0,)),
    #     transforms.RandomErasing(),
    #     transforms.GaussianBlur(15),
    #     torchio.transforms.RandomNoise(std=(0,0.001))

    #     ])
    
    # train_transforms2 = transforms.Compose([
    #     RandomTemporalSubsample(4),
    #     #transforms.Resize(80),
    #     transforms.RandomCrop(80),
    #     transforms.RandomRotation(degrees=(-10, 10), fill=(0,)),
    #     transforms.RandomErasing(),
    #     transforms.GaussianBlur(15),
    #     torchio.transforms.RandomNoise(std=(0,0.001))

    #     ])
    train_transforms = transforms.Compose([
                RandomTemporalSubsample(4),
                #transforms.Resize(80),
                transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40)),
                transforms.GaussianBlur(1),
                torchio.transforms.RandomNoise(std=(0,0.1)),
                transforms.RandomErasing(p=0.5, scale=(0.02,0.2), ratio=(0.3,3)),
            

            ])
    train_transforms2 = train_transforms
            
    test_transform = transforms.Compose([
      UniformTemporalSubsample(4),
      transforms.Resize(80),
    ])

 
    custom_dataset_train = MyDataset(args.og_choose, args.dataset_root,'train',trans=train_transforms,target_list=range(args.classes))

    train_loader = DataLoader(dataset=custom_dataset_train,
       batch_size=args.batch_size,
       num_workers=args.num_workers,
       shuffle=True,trans=train_transforms,trans2=train_transforms2)



    custom_dataset_test = MyDataset(args.og_choose,  args.dataset_root,'test',trans=test_transform,target_list=range(args.classes))

    test_loader = DataLoader(dataset=custom_dataset_test, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,trans=test_transform,test=True)
    print('Training Data: ', len(train_loader(0).dataset))
    print('Test Data: ', len(test_loader(0).dataset))
    init_lr = args.learning_rate * args.batch_size / 64  
    # Choose the model
    if args.model_use == 'ResNet':
        model = generate_model(args.model_depth, n_class)
        model = model.to(device)
        model = model.to(device)

        
        optimizer = optim.RAdam(
            model.parameters(),
            lr= args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.005,
        )

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if  args.checkpoint:
            model, optimizer, lr_schedule, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_schedule)
            args.epochs = args.epochs-epoch    
    elif args.model_use=='Times':
        model = TimeSformer(
        dim = 512,
        image_height = args.img_height,        # image height
        image_width = args.img_width,        # image width  
        patch_height = 20,         # patch height
        patch_width = 20,         # patch width  
        num_frames = args.img_depth,           # number of image frames, selected out of video
        num_classes = n_class,
        depth = 8,
        heads = 4,
        dim_head =  64,
        attn_dropout = 0.1,
        ff_dropout = 0.1
        )
        model = model.to(device)

        
        optimizer = optim.RAdam(
            model.parameters(),
            lr= args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.005,
        )

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if  args.checkpoint:
            model, optimizer, lr_schedule, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_schedule)
            args.epochs = args.epochs-epoch    
     
   
    best_loss = 10000
    args.writer = writer
    for epoch in range(args.epochs +1):
        acc = test(model, device, test_loader, epoch, args) 
        loss_record = train(epoch, model, device, train_loader, optimizer, args)
        is_best = loss_record.avg < best_loss 
        best_loss = max(loss_record.avg, best_loss)
        writer.add_scalar('training loss',
                    loss_record.avg, epoch)
        writer.add_scalar('K means',acc,epoch)
        if is_best:
            torch.save(model.state_dict(), args.model_dir)

if __name__ == '__main__':
    main()
