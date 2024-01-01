import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter
from tqdm import tqdm
import numpy as np
import os
import umap
import matplotlib.pyplot as plt
from models.resnet3d_supervised import ResNet, BasicBlock,generate_model 
from data.highD_loader import MyDataset,ScenarioLoader
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import json
from models.timesformer_supervised import TimeSformer
from einops import rearrange
import torch_optimizer as torchOptim

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def visualize_scenarios(data,flag,args):
    for k in range(2):
        for i in range(args.img_depth):
            if flag==1:
                a = data[k,i,:,:,:]
            else:
                a = data[k,:,i,:,:]

            aa = a.reshape(args.img_height,args.img_width)
            stdName = args.save_dir+'\OG_'+str(k)+'_' +str(i)+'.png'
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

def train(model, train_loader, labeled_eval_loader, optimizer, exp_lr_scheduler, args):
    criterion1 = nn.CrossEntropyLoss() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        #exp_lr_scheduler.step()
        running_loss = 0.0

        if args.UMAP:
            class_vec,feature_vec = model(Variable(torch.randn(2,1,args.img_height,args.img_width,args.img_depth).to(device)))
            represen_x = np.zeros((len(labeled_eval_loader.dataset),feature_vec.shape[1]))
            train_y    = np.zeros((len(labeled_eval_loader.dataset),))
            for batch_idx, (x,  label, idx) in enumerate(tqdm(labeled_eval_loader)):            
                if batch_idx == 0:
                    idx = idx.cpu().data.numpy()
                    label = label.to(device)
                    label = label.cpu().data.numpy()
                    train_y[idx] = label
                    x = x.to(device)
                    if args.model_use=='Times':
                        x = rearrange(x, 'b c f h w  -> b f c h w ')
                    _,represen = model(x)
                    represen = represen.cpu().data.numpy()
                    represen_x[idx,:] = represen
                else:
                    idx = idx.cpu().data.numpy()
                    label = label.to(device)
                    label = label.cpu().data.numpy()
                    train_y[idx] = label
                    if args.model_use=='Times':
                        x = rearrange(x, 'b c f h w  -> b f c h w ')
                    x = x.to(device)
                    _,represen = model(x) 
                    represen = represen.cpu().data.numpy()
                    represen_x[idx,:] =  represen

            U = umap.UMAP(n_components = 2)
            print('Shape_Extracted', represen_x.shape)
            embedding2 = U.fit_transform(represen_x,)
            fig, ax = plt.subplots(1, figsize=(14, 10))
            plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=train_y, cmap='Spectral')
            savename = args.save_dir + r'\UMAP_'+str(epoch)+'.png'
            plt.savefig(savename)

        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            # shape of data (batch x channel x frame x height x width)
            if args.model_use=='Times':
                x = rearrange(x, 'b c f h w  -> b f c h w ')
            if args.viz==True and args.model_use=='Times':
                visualize_scenarios(x,1,args)
            elif args.viz==True:
                visualize_scenarios(x,0,args)            
            x, label = x.to(device), label.to(device)
            output1, _ = model(x)
            loss= criterion1(output1, label)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        args.writer.add_scalar('training loss',
                            running_loss/batch_idx,
                            epoch)
  
        if(epoch%5 == 0):
            save_checkpoint(model, optimizer, args.model_dir, epoch, exp_lr_scheduler)
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        acc = test(model, labeled_eval_loader, epoch,args)

def test(model, test_loader,epoch, args):
    model.eval() 
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        if args.model_use=='Times':
            x = rearrange(x, 'b c f h w  -> b f c h w ')
        x, label = x.to(device), label.to(device)
        output1,  _ = model(x)
        if args.head=='head1':
            output = output1
        _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    args.writer.add_scalar('test acc', acc, epoch) 
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='classific',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int) # 5 works for the clustering task
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_labeled_classes', default=6, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--ssl_dir', type=str, default='./data/experiments/selfsupervised_learning_scenario_barlow_twins/')
    parser.add_argument('--model_name', type=str, default='ssl_labelled')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_use', type=str, default='ResNet')
    parser.add_argument('--ssl_type', type=str, default='Barlow')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--UMAP', type=bool, default=False)
    parser.add_argument('--img_width', type=int, default=120)
    parser.add_argument('--img_height', type=int, default=120)
    parser.add_argument('--img_depth', type=int, default=4)
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--SSL', type=str, default='Barlow')
    parser.add_argument('--split_strategy', type=str, default='equal') # 'equal' or 'unequal'

    args = parser.parse_args()
    args.ssl_dir = args.ssl_dir + args.model_use + '/' + str(args.model_depth) + '/'
    if args.model_use == 'ResNet':
        args.ssl_dir = args.ssl_dir + 'ssl_net_120_resnet.pth'
        args.model_name = args.model_name + '_resnet_' + str(args.model_depth) + '_' +str(args.SSL) + '_' + str(args.num_labeled_classes)
    else:
        args.ssl_dir = args.ssl_dir + 'ssl_net_120_times.pth'
        args.model_name = args.model_name + '_times_' +str(args.SSL) + '_' + str(args.num_labeled_classes)

    
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name, args.model_use, str(args.model_depth))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.save_dir = model_dir
    summar_dir = args.save_dir+'/runs/'+ str(args.model_name)
    if not os.path.exists(summar_dir):
        os.makedirs(summar_dir)

    writer = SummaryWriter(summar_dir)
    args.writer = writer
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name)
    
    # Choose the model
    if args.model_use == 'ResNet': 
        model = generate_model(args.model_depth, args.num_labeled_classes) 
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
       
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)       
        if args.SSL == 'Barlow' or args.SSL == 'Temporal':
            if not args.checkpoint:
                    state_dict = torch.load(args.ssl_dir)
                    if args.ssl_type == "Temporal": 
                       # if args.model_depth == 10:
                        state_dict = state_dict['model_state_dict']
                        del state_dict['module.fc.weight']
                        del state_dict['module.fc.bias']
                    elif args.ssl_type == "Barlow":
                        del state_dict['bn.running_mean']
                        del state_dict['bn.running_var']
                        del state_dict['bn.num_batches_tracked']
                        # Initialize prefix 
                        delProjector = 'projector'
                        # Prefix key match in dictionary
                        for key, val in state_dict.items(): 
                            if key.startswith(delProjector):
                                del key

                    model.load_state_dict(state_dict, strict=False)
                    for name, param in model.named_parameters(): 
                        if 'head' not in name and 'layer4' not in name:
                        #if 'head' not in name:
                            param.requires_grad = False
                        else:
                            print(name)
    
            else:
    
                model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_scheduler)
                args.epochs = args.epochs-epoch

    else:
        model = TimeSformer(
            dim = 512,
            image_height = args.img_height,        # image height
            image_width = args.img_width,        # image width  
            patch_height = 20,         # patch height
            patch_width = 20,         # patch width  
            num_frames = args.img_depth,           # number of image frames, selected out of video
            num_classes = args.num_labeled_classes,
            depth = 8,
            heads = 4,
            dim_head =  64,
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        if args.SSL == 'Barlow' or args.SSL == 'Temporal':
            if not args.checkpoint:
                    state_dict = torch.load(args.ssl_dir)
                    if args.ssl_type == "Temporal": 
                        del state_dict['fc.weight']
                        del state_dict['fc.bias']
                    elif args.ssl_type == "Barlow":
                        del state_dict['bn.running_mean']
                        del state_dict['bn.running_var']
                        del state_dict['bn.num_batches_tracked']
                        # Initialize prefix 
                        delProjector = 'projector'
                        # Prefix key match in dictionary
                        for key, val in state_dict.items(): 
                            if key.startswith(delProjector):
                                del key

                    model.load_state_dict(state_dict, strict=False)
                    for name, param in model.named_parameters(): 
                        if 'head' not in name and 'layers.3' not in name:
                            param.requires_grad = False
                        else:
                            print(name)

    
            else:
    
                model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_scheduler)   

    print("-------------------Model loaded------------------------")
    if args.split_strategy == 'equal':
        labeled_train_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', shuffle=True, aug='once', target_list = range(args.num_labeled_classes))
        labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', shuffle=False,aug=None, target_list = range(args.num_labeled_classes))
    elif args.split_strategy == 'unequal':
        labeled_train_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='trainUE', shuffle=True, aug='once', target_list = range(args.num_labeled_classes))
        labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='testUE', shuffle=False,aug=None, target_list = range(args.num_labeled_classes))        
    print("-------------------Data loaded------------------------")
    print("-------------------Train Start------------------------")
    
    if args.mode == 'train':
        train(model, labeled_train_loader, labeled_eval_loader,optimizer, exp_lr_scheduler, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
    print("-------------------Train End------------------------")
      
    print('test on labeled classes')
    args.head = 'head1'
    acc = test(model, labeled_eval_loader, args.epochs, args)
    result_save_name = args.save_dir + '\Test_'+ args.model_name + '.json'
    values = {'Accuracy':acc,'n_labelled':args.num_labeled_classes}

    with open(result_save_name, 'w') as outfile:
        json.dump(values, outfile)

    print("-------------------Done------------------------")
