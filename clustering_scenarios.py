import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import  PairEnum, cluster_acc, Identity, AverageMeter, seed_torch,BCE
from utils import ramps 
from tqdm import tqdm
from sklearn.ensemble import  RandomTreesEmbedding
import numpy as np
import os
from sklearn.cluster import KMeans
from models.resnet3d_finetune import ResNet, BasicBlock, generate_model 
from data.highD_loader import ScenarioLoader, ScenarioLoaderMix, ScenarioLoaderBuffer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from models.timesformer_finetune import TimeSformer
import random
from sklearn.metrics.pairwise import pairwise_distances
import umap
from einops import rearrange
from tensorboardX import SummaryWriter
import json
from sklearn.neighbors import kneighbors_graph
from scipy.io import loadmat
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 
def visualize_scenarios(data,flag,args):
    data = data.cpu().numpy()
    for k in range(5):
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

def debug_cluster(data,labels):
    kmeans = KMeans(n_clusters=args.num_unlabeled_classes,n_init=20).fit(data)         
    y = kmeans.labels_
    acc, nmi, ari = cluster_acc(labels.astype(int), y.astype(int)),nmi_score(labels.astype(int), y.astype(int)), ari_score(labels.astype(int), y.astype(int)) 
    print('K means acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

def divide_unknown_known(model,train_loader,num_lab,num_unlab,args):
    model.eval()
    unknown_represe = np.zeros((num_unlab,512))
    unknown_label    = np.zeros((num_unlab,))
    known_represe = np.zeros((num_lab,512))
    known_label = np.zeros((num_lab,))
    for batch_idx, ((x, x_bar),  label, idx) in enumerate((train_loader)):
        if args.model_use=='Times':
            x = rearrange(x, 'b c f h w  -> b f c h w ')
            x_bar = rearrange(x_bar, 'b c f h w  -> b f c h w ')

         
        mask_rf = idx<num_lab # label<args.num_labeled_classes
        
        # Unknown
        Y_test = (label[~mask_rf]).detach()
        ulb_data = (x[~mask_rf]).detach()
        idx_ulb  = (idx[~mask_rf]).detach()
        idx_ulb  = idx_ulb.cpu().data.numpy()
        idx_ulb  = idx_ulb - num_lab
        idx_ulb  = idx_ulb.astype(int) 
        
        # Known
        Y_train = (label[mask_rf]).detach()
        lb_data = (x[mask_rf]).detach()
        idx_lb  = (idx[mask_rf]).detach()
        idx_lb  = idx_lb.cpu().data.numpy()
        idx_lb  = idx_lb.astype(int)             

        # Unknown
        if ulb_data.shape[0]>0:
            unknown_label[idx_ulb] = Y_test
            ulb_data = ulb_data.to(device)
            _,_,represen = model(ulb_data) 
            represen = represen.cpu().data.numpy()
            unknown_represe[idx_ulb,:] = represen
        
        # Known
        if lb_data.shape[0]>0: 
            known_label[idx_lb] = Y_train
            lb_data = lb_data.to(device)
            _,_,rep_tr_known = model(lb_data) 
            rep_tr_known = rep_tr_known.cpu().data.numpy()
            known_represe[idx_lb,:] = rep_tr_known
    return unknown_represe, unknown_label, known_represe, known_label  


def other_similarities(model,train_loader,num_labelled,num_unlabelled,args):
    with torch.no_grad():
        unknown_represe, unknown_label, known_represe, known_label = divide_unknown_known(model,train_loader,num_labelled,num_unlabelled,args)          
        if args.similarity == 'cosine':
            D = pairwise_distances(unknown_represe, metric="cosine")
        elif args.similarity == 'l2':
            D = pairwise_distances(unknown_represe, metric="l2")
            row_sums = D.sum(axis=1)
            new_matrix = D / row_sums[:, np.newaxis]
            D = new_matrix
        elif args.similarity == 'knn':
            S = kneighbors_graph(unknown_represe, 5, mode='connectivity', include_self=True)
            S = S.toarray()
            return S, unknown_represe, unknown_label, known_represe, known_label
        
    S = 1 - D
    return S, unknown_represe, unknown_label, known_represe, known_label  

def train_RF(model,train_loader,num_lab,num_unlab,args):
    with torch.no_grad():

        unknown_represe, unknown_label, known_represe, known_label = divide_unknown_known(model,train_loader,num_lab,num_unlab,args)          
        model_rf = RandomTreesEmbedding(n_estimators=args.num_trees, n_jobs=-1,max_depth=None).fit(unknown_represe)
        print('Trees_Trained')
        model_rf.index(type_expect = 1)
        rfap = model_rf.encode_rfap(unknown_represe)    
        D = pairwise_distances(rfap, metric="hamming")
    S = 1 - D 
    return S, unknown_represe, unknown_label, known_represe, known_label
  

                
def plot_umap(data, label,epoch,args):
    U = umap.UMAP(n_components = 2)
    print('Shape_Extracted', data.shape)
    embedding2 = U.fit_transform(data,)
    classesList = np.unique(label)
    classes = [str(i) for i in classesList]   
    fig, ax = plt.subplots(1)
    sc = ax.scatter(embedding2[:,0], embedding2[:,1], s=2, c=label, cmap='jet', alpha=1.0)
    ax.set(xticks=[], yticks=[])
    cbar = plt.colorbar(sc, ticks=classesList)#, boundaries=np.arange(classesList[0],classesList[-1]+1)-0.5)
    saveName = args.save_dir+r'\Scenario_'+str(epoch)+'.png'
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.grid(False)
    plt.savefig(saveName)
    plt.close()
    
    
def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    # optimizer = torchOptim.RAdam(
    #     model.parameters(),
    #     lr= args.lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=args.weight_decay,
    # )
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    if args.similarity is not 'rank':
        criterion2 = nn.BCELoss()
    else:
        criterion2 = BCE()
    for epoch in range(args.epochs): 
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        num_unlabelled = len(unlabeled_eval_loader.dataset)
        num_labelled = len(train_loader.dataset) - num_unlabelled      
        # Train RF
        if args.similarity is not 'rank':
            if args.similarity == 'RF':
                S, unknown_represe, unknown_label, known_represe, known_label = train_RF(model,train_loader,num_labelled,num_unlabelled,args)
            else:
                S, unknown_represe, unknown_label, known_represe, known_label = other_similarities(model,train_loader,num_labelled,num_unlabelled,args)
        
        if args.debug_viz: # UMAP+K means
            debug_cluster(unknown_represe,unknown_label)
            plot_umap(unknown_represe, unknown_label,epoch,args)
        running_loss = 0.0
        # Iterative optimisation    
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):         
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            label = label.long()
            mask_lb = label < args.num_labeled_classes
            
            if args.model_use=='Times':
                x = rearrange(x, 'b c f h w  -> b f c h w ')
                x_bar = rearrange(x_bar, 'b c f h w  -> b f c h w ')
            if args.viz==True and args.model_use=='Times':
                visualize_scenarios(x,1,args)
            elif args.viz==True:
                visualize_scenarios(x[~mask_lb],0,args) 
                
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)

            if  torch.isnan(output1).any() or torch.isnan(output2).any():
                print('Found NaN value')

            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)            
            rank_feat = (feat[~mask_lb]).detach()
            if len(rank_feat)  == 0:
                continue  
            if args.similarity == 'rank':
                rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                rank_idx1, rank_idx2 = PairEnum(rank_idx)
                rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
                
                rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                rank_idx2, _ = torch.sort(rank_idx2, dim=1)

                rank_diff = rank_idx1 - rank_idx2
                rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                target_ulb = torch.ones_like(rank_diff).float().to(device)
                target_ulb[rank_diff > 0] = -1

                prob1_ulb, _ = PairEnum(prob2[~mask_lb])
                _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

                if args.use_labelled_classes:
                    loss_ce = criterion1(output1[mask_lb], label[mask_lb])

                    label[~mask_lb] = (output2[~mask_lb]).detach().max(1)[1] + args.num_labeled_classes

                    loss_ce_add = w * criterion1(output1[~mask_lb], label[~mask_lb]) / args.rampup_coefficient * args.increment_coefficient
                    loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
                    consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
                    loss = loss_ce + loss_bce + loss_ce_add + w * consistency_loss                
                else:
                    loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
                    consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
                    loss =  loss_bce  + w * consistency_loss

            else:
    
                prob1_ulb, _ = PairEnum(prob2[~mask_lb])
                _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])     
    
                loss_ce = criterion1(output1[mask_lb], label[mask_lb])
                label[~mask_lb] = (output2[~mask_lb]).detach().max(1)[1] + args.num_labeled_classes
                loss_ce_add = w * criterion1(output1[~mask_lb], label[~mask_lb]) / args.rampup_coefficient *  args.increment_coefficient       
                x1, x2  = PairEnum(rank_feat)
                x1 = x1.cpu().data.numpy()
                x2 = x2.cpu().data.numpy()
                idx = (idx[~mask_lb]).detach() 
                idx_ulb1 = idx.reshape(-1,1)
                idx1, idx2 = PairEnum(idx_ulb1)
                idx1 = idx1.cpu().data.numpy()
                idx2 = idx2.cpu().data.numpy()
                idx1 = idx1.astype(int)
                idx2 = idx2.astype(int)
                idx2 = idx2 - num_labelled          
                idx1 = idx1 - num_labelled
                similarity_drf = np.zeros((x1.shape[0],))
                for sima in (range(similarity_drf.shape[0])):
                    similarity_drf[sima] = S[idx1[sima],idx2[sima]]
                similarity_drf = torch.from_numpy(similarity_drf).float().to(device)               
                P = prob1_ulb.mul_(prob2_ulb)
                P = P.sum(1)            
                loss_bce = criterion2(P,similarity_drf)                       
                consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
               # print('CE loss {}, BCE loss {}, CE Additional {}, Consistency loss {}'.format(loss_ce, loss_bce, loss_ce_add, consistency_loss))
                if args.use_labelled_classes:
                    loss = loss_ce + loss_bce + loss_ce_add + w * consistency_loss
                else:
                    loss =  loss_bce + w * consistency_loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        args.writer.add_scalar('training loss',
                            loss_record.avg,
                            epoch)        
        if(epoch%5 == 0):
            save_checkpoint(model, optimizer, args.model_dir, epoch, exp_lr_scheduler)   
            
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        acc_labelled = test(model, labeled_eval_loader, args)
        print('test on unlabeled classes')
        args.head='head2'
        acc_unlabelled = test(model, unlabeled_eval_loader, args)
        args.writer.add_scalar('Unlablled Test Acc',
                            acc_unlabelled,
                            epoch)
        args.writer.add_scalar('Lablled Test Acc',
                            acc_labelled,
                            epoch) 
        exp_lr_scheduler.step()
        print('Current learning rate is {}'.format(get_lr(optimizer)))
    return model

def test(model, test_loader, args):
    preds=np.array([])
    targets=np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
            x, label = x.to(device), label.to(device)
            if args.model_use=='Times':
                x = rearrange(x, 'b c f h w  -> b f c h w ') 
            output1, output2, _ = model(x)
            if args.head=='head1':
                output = output1
            else:
                output = output2
                #label = (label - args.num_labeled_classes).long()
            _, pred = output.max(1)
            targets=np.append(targets, label.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())
        acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
        return acc


def collectClusterLabels(model, unlabeled_loader, args):
    cluster_features = np.zeros((len(unlabeled_loader.dataset),512))
    cluster_labels = np.zeros((len(unlabeled_loader.dataset),))
    cluster_true_labels = np.zeros((len(unlabeled_loader.dataset),))
    args.head2 = 'head2'
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(tqdm(unlabeled_loader)):
            x = x.to(device)
            if args.model_use=='Times':
                x = rearrange(x, 'b c f h w  -> b f c h w ') 
            output1, output2, features = model(x)
            if args.head=='head1':
                output = output1
            else:
                output = output2
            _, pred = output.max(1)
            features.detach()
            strtId = batch_idx * args.batch_size
            endID = (batch_idx*args.batch_size)+features.shape[0]
            #if endID>(len(unlabeled_loader)*args.batch_size):
            #    endId =len(unlabeled_loader)*args.batch_size
            cluster_features[strtId:endID,:] = features.cpu().numpy()
            cluster_labels[strtId:endID,] = pred.cpu().numpy()
            cluster_true_labels[strtId:endID,] = label.cpu().numpy()
    return cluster_features,cluster_labels,cluster_true_labels


def cluster_label(model, test_loader, args):
    preds=np.array([])
    targets=np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
            x, label = x.to(device), label.to(device)
            if args.model_use=='Times':
                x = rearrange(x, 'b c f h w  -> b f c h w ') 
            output1, output2, _ = model(x)
            if args.head=='head1':
                output = output1
            else:
                output = output2
            _, pred = output.max(1)
            preds=np.append(preds, pred.cpu().numpy())
    return preds

# TO DO
# exemplary set for the labelled data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.01) # 0.01
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=80, type=int) # 80 
    parser.add_argument('--unknown_class_list', default=None , type=list) 
    parser.add_argument('--rampup_coefficient', type=float, default=50) #50
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=170 , type=int) #170
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_unlabeled_classes', default=7, type=int)
    parser.add_argument('--num_labeled_classes', default=6, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='ssl_labelled')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: HighD, round')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_use', type=str, default='ResNet')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--debug_viz', type=bool, default=True)
    parser.add_argument('--img_width', type=int, default=120)
    parser.add_argument('--img_height', type=int, default=120)
    parser.add_argument('--img_depth', type=int, default=4)
    parser.add_argument('--num_trees', type=int, default=500)
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_root', type=str, default='./data/experiments/supervised_learning_scenarios/')
    parser.add_argument('--osr_root', type=str, default='./data/experiments/open_set_recognition/') 
    parser.add_argument('--SSL', type=str, default='Barlow')
    parser.add_argument('--OSR_method', type=str, default='EVT')
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--similarity', type=str, default='RF')
    parser.add_argument('--cluster_mode', type=str, default='buffer') # can be either single_shot or buffer based
    parser.add_argument('--split_strategy', type=str, default='unequal') # 'equal' or 'unequal'
    parser.add_argument('--buffer_select', type=int, default=32,help='used to select the buffer stored can be fixed sized buffer or \
                        complete buffer with infinite storage')
    parser.add_argument('--clustering_cycle', type=int, default=1)
    parser.add_argument('--clustered_classes', type=int, default=None)
    parser.add_argument('--use_labelled_classes', type=bool, default=True)
    parser.add_argument('--clustering_method', type=str, default='IO', help='Interative optimisation IO or AE')
    
    
    args = parser.parse_args()
    seed_torch(args.seed)
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name, args.model_use, str(args.model_depth))
    args.model_root = os.path.join(args.model_root, args.model_use, str(args.model_depth))
    args.osr_root = os.path.join(args.osr_root, args.model_use)
    args.osr_root = args.osr_root + '/' + str(args.model_depth) 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


        
    if args.model_use == 'ResNet':
        args.model_name = args.model_name + '_resnet_' + str(args.model_depth) + '_' +str(args.SSL)+ '_' + str(args.num_labeled_classes)
    else:
        args.model_name = args.model_name + '_times_' +str(args.SSL)+ '_' + str(args.num_labeled_classes)
        
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name)   
    args.cluster_model_dir = model_dir+'/'+'{}_cluster.pth'.format(args.model_name + '_' + str(args.clustered_classes)+ '_' + str(args.num_unlabeled_classes))   
    args.mat_load_dir = args.model_name + '_' + str(args.clustered_classes) + '_' + str(args.num_unlabeled_classes)             
    model_dir= os.path.join(args.exp_root, runner_name, args.model_use, str(args.model_depth))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.save_dir = model_dir

    summar_dir = args.save_dir+'/runs/'+ str(args.model_name)
    if not os.path.exists(summar_dir):
        os.makedirs(summar_dir)
    writer = SummaryWriter(summar_dir)     
    args.model_root = args.model_root+'/'+'{}.pth'.format(args.model_name)
    matfileToLoad = args.mat_load_dir + '_' + str(args.buffer_select) + '_' 
    mat_file_name = args.osr_root+'/'+'{}.mat'.format(matfileToLoad)  
    
    elbow_value = loadmat(mat_file_name)['elbow']
    elbow_value = np.array(elbow_value[0][0])
    print('Model loaded from {}'.format(args.model_root))
    elbow_value = args.num_unlabeled_classes
    args.num_unlabeled_classes = elbow_value
    args.writer = writer
    # Choose the model
    if args.model_use == 'ResNet': 
        model = generate_model(args.model_depth, args.num_labeled_classes,args.num_unlabeled_classes) 
        model = model.to(device)
        if args.use_labelled_classes:
            if not args.checkpoint:   
                state_dict = torch.load(args.model_root)
                model.load_state_dict(state_dict, strict=False)
                for name, param in model.named_parameters(): 
                    if 'head' not in name and 'layer4' not in name:
                        param.requires_grad = False
                    else:
                        print(name)
            else:

                model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir, lr_scheduler)
                args.epochs = args.epochs-epoch
        else:
            state_dict = torch.load('E:\Lakshman\Final_Assembly\data\experiments\selfsupervised_learning_scenario_barlow_twins\ResNet\\18\ssl_net_120_resnet.pth')
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
        model = TimeSformer(
            dim = 512,
            image_height = args.img_height,        # image height
            image_width = args.img_width,        # image width  
            patch_height = 20,         # patch height
            patch_width = 20,         # patch width  
            num_frames = args.img_depth,           # number of image frames, selected out of video
            num_classes = args.num_labeled_classes,
            num_Unlabelledclasses = args.num_unlabeled_classes, 
            depth = 8,
            heads = 4,
            dim_head =  64,
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )
        model = model.to(device)
     
        if not args.checkpoint:
            state_dict = torch.load(args.model_root)
            model.load_state_dict(state_dict, strict=False)
            for name, param in model.named_parameters(): 
                if 'head' not in name and 'layers.3' not in name:
                    param.requires_grad = False
                else:
                    print(name)
        else:
            model, optimizer, exp_lr_scheduler, epoch = load_checkpoint(model, optimizer, args.model_dir,lr_scheduler)
            args.epochs = args.epochs-epoch 
            
    # already clustered classes exist
    if args.clustered_classes is None:
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        unknown_classses = list(range(args.num_labeled_classes, num_classes))
        #random.shuffle(unknown_classses)    
        if args.unknown_class_list is not None:
            unknown_classses = args.unknown_class_list
        else:
            unknown_classses = unknown_classses[:args.num_unlabeled_classes]
        assert len(unknown_classses) == args.num_unlabeled_classes

        known_classes = list(range(args.num_labeled_classes)) 
        classes_chosen = known_classes + unknown_classses
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    else:
        num_classes = args.num_labeled_classes +args.clustered_classes + args.num_unlabeled_classes
        knownclassesNumber = args.num_labeled_classes+args.clustered_classes
        unknown_classses = list(range(knownclassesNumber, num_classes))
        #random.shuffle(unknown_classses)    
        if args.unknown_class_list is not None:
            unknown_classses = args.unknown_class_list
        else:
            unknown_classses = unknown_classses[:args.num_unlabeled_classes]
        assert len(unknown_classses) == args.num_unlabeled_classes
        
        known_classes = list(range(args.num_labeled_classes))
        classes_chosen = known_classes + unknown_classses
        
    if args.cluster_mode == 'single_shot':
        if args.split_strategy == 'equal':
            mix_train_loader = ScenarioLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', shuffle=True, aug='twice', labeled_list=known_classes, unlabeled_list=unknown_classses)
            unlabeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=False, target_list = unknown_classses)
            unlabeled_eval_loader_test = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = unknown_classses)
            labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = known_classes)
            all_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = classes_chosen)
        elif args.split_strategy == 'unequal':
            mix_train_loader = ScenarioLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='trainUE', shuffle=True, aug='twice', labeled_list=known_classes, unlabeled_list=unknown_classses)
            unlabeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='trainUE', aug='once', shuffle=False, target_list = unknown_classses)
            unlabeled_eval_loader_test = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='testUE', aug=None, shuffle=False, target_list = unknown_classses)
            labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='testUE', aug=None, shuffle=False, target_list = known_classes)
            all_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='testUE', aug=None, shuffle=False, target_list = classes_chosen)
            
    elif args.cluster_mode == 'buffer':
        mix_train_loader = ScenarioLoaderBuffer(root=args.dataset_root, unknown_root=mat_file_name, batch_size=args.batch_size, split='unknownTrain', aug='twice', shuffle=True, labeled_list=known_classes,sizeMatch=True)      
        unlabeled_eval_loader = ScenarioLoader(root=mat_file_name, batch_size=args.batch_size, split='unknownTrain', aug='once', shuffle=False)
        unlabeled_eval_loader_test = ScenarioLoader(root=mat_file_name, batch_size=args.batch_size, split='unknownTest', aug=None, shuffle=False)
        labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = known_classes)
        all_eval_loader = ScenarioLoader(root=args.dataset_root, unknown_root=mat_file_name, batch_size=args.batch_size, split='testBuffer', aug=None, shuffle=False, target_list = known_classes)  # known_classes because the rest will be from buffer      
    
    if args.mode == 'train':
        if args.model_use == 'ResNet':
            if args.clustered_classes is not None:
                num_classes -= args.clustered_classes 
            save_weight = model.head1.weight.data.clone()
            save_bias = model.head1.bias.data.clone()
            model.head1 = nn.Linear(512, num_classes).to(device)
            model.head1.weight.data[:args.num_labeled_classes] = save_weight
            model.head1.bias.data[:] = torch.min(save_bias) - 1.
            model.head1.bias.data[:args.num_labeled_classes] = save_bias 
        else:
            if args.clustered_classes is not None:
                num_classes -= args.clustered_classes 
            save_weight_0 = model.head1[0].weight.data.clone()
            save_bias_0 = model.head1[0].bias.data.clone()
            save_weight_1 = model.head1[1].weight.data.clone()
            save_bias_1 = model.head1[1].bias.data.clone()
            model.head1[1] = nn.Linear(512, num_classes).to(device)
            model.head1[1].weight.data[:args.num_labeled_classes] = save_weight_1
            model.head1[1].bias.data[:] = torch.min(save_bias_1) - 1.
            model.head1[1].bias.data[:args.num_labeled_classes] = save_bias_1            
            
        model_trained = train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)
        torch.save(model_trained.state_dict(), args.cluster_model_dir)
        print("model saved to {}.".format(args.cluster_model_dir))
    else:
        print("model loaded from {}.".format(args.model_dir))
        model.head1 = nn.Linear(512, num_classes).to(device)
        model.load_state_dict(torch.load(args.cluster_model_dir))


    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    acc_labelled = test(model_trained, labeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    acc_unlabelled_test = test(model_trained, unlabeled_eval_loader_test, args)
    #print('test on all classes (test split)')
    #test(model, all_eval_loader, args)
    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    acc_unlabelled_train = test(model_trained, unlabeled_eval_loader, args)
    cluster_targets = cluster_label(model_trained, unlabeled_eval_loader, args)
    
    cluster_features, cluster_labels, cluster_true_labels = collectClusterLabels(model_trained, unlabeled_eval_loader, args)
    
    cluster_data_save = {'features': cluster_features, 'labels': cluster_labels, 'true_labels': cluster_true_labels}
    plot_umap(cluster_features, cluster_true_labels, 1000, args)
    clusterSaveDir = args.save_dir+'\\cluster'+ '_' + args.SSL + '_' + str(args.num_labeled_classes) + str(args.clustered_classes)\
                        + '_' + str(args.num_unlabeled_classes) + '_' + str(args.clustering_cycle) + '_'
    np.save(clusterSaveDir, cluster_data_save)
    
    result_save_dir = args.save_dir + '\\' +  args.model_name + '_' + args.SSL + '_' + str(args.num_labeled_classes) + str(args.clustered_classes) + '_' + str(args.num_unlabeled_classes)+\
        '_' + str(args.clustering_cycle)+'.json'
    cluster_data = {'Test Unlab CA':acc_unlabelled_test,'Train Unlab CA':acc_unlabelled_train,'Train Lab CA':acc_labelled,'Targets':list(cluster_targets)}
    with open(result_save_dir, 'w') as outfile:
        json.dump(cluster_data, outfile)
                 