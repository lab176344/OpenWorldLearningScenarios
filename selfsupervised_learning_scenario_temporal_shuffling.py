from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import  transforms
import os
import os.path
import numpy as np
from data.highDSSl_loader_temporal_shuffling import MyDataset,DataLoader
from utils.util import AverageMeter, accuracy
from tqdm import tqdm
from models.resnet3d import generate_model 
from models.timesformer import TimeSformer
import math
import matplotlib.pyplot as plt
from einops import rearrange
from tensorboardX import SummaryWriter
from utils.cosine_annealing import CosineAnnealingWarmupRestarts
import torch.distributed as dist
import torch.multiprocessing as mp


def visualize_scenarios(data,flag,args):
    for k in range(2):
        for i in range(args.img_depth):
            if flag==1:
                a = data[k,i,:,:,:]
            else:
                a = data[k,:,i,:,:]
            aa = a.reshape(args.img_height,args.img_width)
            stdName = args.save_dir+'\OG_'+str(k)+'_' +str(i)+'.pdf'
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

def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion,args,dist=None,rank=0):
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    model.train()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader(epoch))):
        # shape of data (batch x channel x frame x height x width)
        if args.model_use=='Times':
            data = rearrange(data, 'b c f h w  -> b f c h w ')

        if args.viz==True and args.model_use=='Times':
            visualize_scenarios(data,1,args)
        elif args.viz==True:
            visualize_scenarios(data,0,args)
            
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True) # add this line
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
        # measure accuracy and record loss
        acc = accuracy(output, label)
        if args.num_gpus>1:
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            dist.all_reduce(acc, dist.ReduceOp.SUM)
            
        acc_record.update(acc[0].item(), data.size(0))
        loss_record.update(loss.item(), data.size(0))


    if rank==0:
        print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

    return loss_record.avg

def test(model, device, dataloader,criterion, args):
    acc_record = AverageMeter()
    model.eval()
    total_loss = 0.0
    for batch_idx, (data, label) in enumerate(tqdm(dataloader())):
        
        if args.model_use=='STAM' or args.model_use=='Times':
            data = rearrange(data, 'b c f h w  -> b f c h w ')
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True) # add this line
        output = model(data)
        loss = criterion(output, label)
        total_loss += loss.item()
 
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))
    avg_loss = total_loss / len(dataloader())

    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record,avg_loss 

def _train(model, train_loader,test_loader, optimizer,criterion,exp_lr_scheduler,args):
    #TODO merge with _train
    writer = SummaryWriter(args.summar_dir)
    world_size = args.num_gpus
    epochs = args.epochs
    device = args.device
    model.to(device)
    model.train()
    for epoch in range(args.epochs +1):
        loss_record = train(epoch, model, device, train_loader, optimizer, exp_lr_scheduler, criterion, args)
        acc_record,avg_loss = test(model, device, test_loader,criterion, args)
        exp_lr_scheduler.step()        
        is_best = acc_record.avg > best_acc 
        best_acc = max(acc_record.avg, best_acc)
            
        writer.add_scalar('training loss',
                        loss_record,
                        epoch)
        if is_best:
            torch.save(model.state_dict(), args.model_dir)
            save_checkpoint(model, optimizer, args.model_dir, epoch, exp_lr_scheduler)  
              
def _train_dist(model, train_loader, optimizer,criterion,exp_lr_scheduler,args):
    assert dist.is_nccl_available()
    #TODO merge with _train
    writer = SummaryWriter(args.summar_dir)
    world_size = args.num_gpus
    epochs = args.epochs
    pbar = tqdm(total=int(epochs * len(train_loader.dataset) /
                            (args.batch_size)),
                desc="init training...".center(50))

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)

    device = torch.device(rank)
    model.to(device)
    if args.batch_size < 4:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model, broadcast_buffers=False, device_ids=[rank])
    model.train()
    
    for epoch in range(args.epochs +1):
        loss_record = train(epoch, model, device, train_loader, optimizer, exp_lr_scheduler, criterion, args, dist, rank)
        if rank==0:
            acc_record,avg_loss = test(model, device, test_loader,criterion, args)
            exp_lr_scheduler.step()        
            is_best = acc_record.avg > best_acc 
            best_acc = max(acc_record.avg, best_acc)
                
            writer.add_scalar('training loss',
                            loss_record,
                            epoch)
            if is_best:
                torch.save(model.state_dict(), args.model_dir)
                save_checkpoint(model, optimizer, args.model_dir, epoch, exp_lr_scheduler) 
def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    # Training settings
    parser = argparse.ArgumentParser(description='Rot_resNet')
    parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                                    help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='ssl_net')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--og_choose', type=list, default=[0,3,6,9])
    parser.add_argument('--model_use', type=str, default='ResNet') #STAM
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--img_width', type=int, default=80)
    parser.add_argument('--img_height', type=int, default=80)
    parser.add_argument('--img_depth', type=int, default=4)
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()
  
    if args.model_use == 'ResNet':
        args.model_name = args.model_name + '_resnet'
    else:
        args.model_name = args.model_name + '_stam'


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

    args.summar_dir = summar_dir
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])  

    custom_dataset_train = MyDataset(args.og_choose, args.dataset_root,'train',trans=train_transforms)

    train_loader = DataLoader(dataset=custom_dataset_train,
       batch_size=args.batch_size,
       num_workers=args.num_workers,
       shuffle=True,trans=train_transforms)

    train_loader = DataLoader(dataset=custom_dataset_train, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers,trans=train_transforms)

    custom_dataset_test = MyDataset(args.og_choose,  args.dataset_root,'test',trans=train_transforms)
    
    test_loader = DataLoader(dataset=custom_dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=False,trans=train_transforms)

    # Choose the model
    if args.model_use == 'ResNet':
        model = generate_model(args.model_depth, n_class)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)    
        if  args.checkpoint:
            model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_scheduler)
            args.epochs = args.epochs-epoch   
    elif args.model_use=='Times':
        model = TimeSformer(
        dim = 256,
        image_height = args.img_height,        # image height
        image_width = args.img_width,        # image width  
        patch_height = 10,         # patch height
        patch_width = 20,         # patch width  
        num_frames = args.img_depth,           # number of image frames, selected out of video
        num_classes = n_class,
        depth = 4,
        heads = 6,
        dim_head =  64,
        attn_dropout = 0.1,
        ff_dropout = 0.1
        )
        
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)    
        if  args.checkpoint:
            model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_scheduler)
            args.epochs = args.epochs-epoch   
    else:
        model = STAM(
                dim = 512,
                image_height = args.img_height,         # image height
                image_width = args.img_width,           # image width  
                patch_height = 10,                       # patch height
                patch_width = 20,                       # patch width  
                num_frames = args.img_depth,            # number of image frames, selected out of video
                space_depth = 2,                       # depth of vision transformer
                space_heads = 8,                        # heads of vision transformer
                space_mlp_dim = 512,                   # feedforward hidden dimension of vision transformer
                time_depth = 2,                         # depth of time transformer (in paper, it was shallower, 6)
                time_heads = 4,                         # heads of time transformer
                time_mlp_dim = 512,                    # feedforward hidden dimension of time transformer
                num_classes = n_class,                  # number of output classes
                space_dim_head = 64,                    # space transformer head dimension
                time_dim_head = 64,                     # time transformer head dimension
                dropout = 0.1,                           # dropout
                emb_dropout = 0.1                        # embedding dropout
                )
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)    
        if  args.checkpoint:
            model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_scheduler)
            args.epochs = args.epochs-epoch   
    criterion = nn.CrossEntropyLoss()
    best_acc = 0 
    args.device = device
    if args.num_gpus > 1:
        mp.spawn(_train_dist,
                    nprocs=args.num_gpus,
                    args=(model, train_loader,test_loader, optimizer,criterion,exp_lr_scheduler,args))
    else:
        _train(model, train_loader,test_loader, optimizer,criterion,exp_lr_scheduler,args)




    # for epoch in range(args.epochs +1):
    #     loss_record = train(epoch, model, device, train_loader, optimizer, exp_lr_scheduler, criterion, args)
    #     acc_record,avg_loss = test(model, device, test_loader,criterion, args)
    #     exp_lr_scheduler.step()        
    #     is_best = acc_record.avg > best_acc 
    #     best_acc = max(acc_record.avg, best_acc)
          
    #     writer.add_scalar('training loss',
    #                    loss_record,
    #                    epoch)
    #     if is_best:
    #         torch.save(model.state_dict(), args.model_dir)
    #         save_checkpoint(model, optimizer, args.model_dir, epoch, exp_lr_scheduler)

if __name__ == '__main__':
    main()
