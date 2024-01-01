from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.utils.linear_assignment_ import linear_assignment
import random
import os
import argparse
import seaborn as sb
from torch import  optim
from PIL import Image, ImageOps, ImageFilter
import umap
from sklearn.cluster import KMeans

#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if(np.isnan(val)):
            print('Val is Nan')
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class GaussianBlur(object):
    def __init__(self, p, seed=1):
        self.seed = seed
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.1 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class RandomCrop(object):
    def __init__(self):
        self.x = 30
        self.y = 15
        self.cut_x = random.randint(0,120)
        self.cut_y = random.randint(0,30)


    def __call__(self, img):
        pass

class Solarization(object):
    def __init__(self, p, seed=1):
        self.p = p
        self.seed = seed
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def plot_umap(data, label, epoch, num_unlabeled_classes, args):
    U = umap.UMAP(n_components = 2,n_neighbors=30)
    print('Shape_Extracted', data.shape)
    embedding2 = U.fit_transform(data)
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=label, cmap='Spectral')
    savename = args.save_dir + r'\UMAP_'+str(epoch)+'.png'
    cbar = plt.colorbar(boundaries=np.arange(num_unlabeled_classes+1)-0.5)
    cbar.set_ticks(np.arange(num_unlabeled_classes))
    plt.savefig(savename)
    plt.close()
    plt.clf()
    plt.cla()


def debug_cluster(data,labels,num_unlabeled_classes):
    kmeans = KMeans(n_clusters=num_unlabeled_classes,n_init=20).fit(data)
    y = kmeans.labels_
    acc = cluster_acc(labels.astype(int), y.astype(int))
    print('K means acc {:.4f}'.format(acc))
    return acc

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_twin_loss(out_1, out_2,device,save_dir,lambda_param=5e-3,visualize = True,epoch=0,eps=0.000001, args=None):
    # normalize the representations along the batch dimension
    #out_1_norm = (out_1 - out_1.mean(dim=0)) / (out_1.std(dim=0)+eps)
    #out_2_norm = (out_2 - out_2.mean(dim=0)) / (out_2.std(dim=0)+eps)
    batch_size = out_1.size(0)
    D = out_1.size(1)
    # cross-correlation matrix
    #c = out_1.T @ out_2
    c = torch.mm(out_1.T, out_2) / batch_size
    #print(c.shape)
    if False and epoch%100==0:
        ans = sb.heatmap(c.cpu().data.numpy(), cmap="Blues")
        fig_name = save_dir + r'\correlations_'+str(epoch)+'.png'
        plt.savefig(fig_name)
        plt.clf()
        plt.cla()
        plt.close()
    # loss
    device = out_1.get_device()
    c_diff = (c - torch.eye(D,device=device)).pow(2) # DxD
    # multiply off-diagonal elems of c_diff by lambda
    c_diff[~torch.eye(D, dtype=bool)] *= lambda_param
    loss = c_diff.sum()

    return loss

def barlow_twin_loss_v2(z1, z2,device,save_dir,lambda_param=5e-3,visualize = True,epoch=0, args=None):
    # empirical cross-correlation matrix
    c = (z1).T @ (z2)
    batch_size = z1.size(0)
    c.div_(batch_size)
    D = z1.size(1)
    #print(c.shape)
    # loss
    c_diff = (c - torch.eye(D,device=device)).pow(2) # DxD
    # multiply off-diagonal elems of c_diff by lambda
    c_diff[~torch.eye(D, dtype=bool)] *= lambda_param
    loss = c_diff.sum()
    if False and epoch%100==0:
        ans = sb.heatmap(c.cpu().data.numpy(), cmap="Blues")
        fig_name = save_dir + r'\correlations_'+str(epoch)+'.png'
        plt.savefig(fig_name)
        plt.clf()
        plt.cla()
    return loss


class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()
    
def min_distance_times_one_minus_softmin(input_dict):
    return (input_dict['distances'] * (1 - torch.softmax(-input_dict['distances'], dim=1)))

def min_one_minus_softmin(input_dict):
        return (1 - torch.softmax(-input_dict['distances'], dim=1))
    
def min_distance(input_dict):
    return input_dict['distances'].min(dim=1)[1] 
   
# if you are changing/editing the model class, create new file
class CACHead(nn.Module):
    def __init__(self, num_classes, centers_lambda=10, learnt_centers=False, device=0):
        super().__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.eye(num_classes).to(device).double() * centers_lambda, requires_grad=learnt_centers)

    def forward(self, x):  # batch_size x 3 x 640 x 480
        distances = torch.sqrt(torch.square(
            x.unsqueeze(1).expand(x.size(0), *self.centers.shape).double() -
            self.centers.unsqueeze(0).expand(x.size(0), *self.centers.shape)
        ).sum(dim=2))  # batch_size, num_classes_or_heads
        output_dict = dict(
            heads=x,
            distances=distances,
        )
        return output_dict
    
class CACLossV1:
    def __init__(self, alw, num_classes, new_class=False):
        self.alw = alw
        self.num_classes = num_classes
        self.new_class = new_class
        self.initial_loss_dict = dict(
            anchor_loss=0,
            tuplet_loss=0,
            total_loss=0
        )

    def __call__(self, outputs, targets):
        distances = outputs['distances']
        true = torch.gather(distances, 1, targets.view(-1, 1)).view(-1)
        non_gt = torch.Tensor(
            [[i for i in range(self.num_classes) if targets[x] != i] for x in range(len(distances))]).long().cuda()
        others = torch.gather(distances, 1, non_gt)

        anchor = self.alw * torch.mean(true)

        tuplet = torch.exp(-others + true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

        total = anchor + tuplet

        output_dict = dict(
            total_loss=total,
            tuplet_loss=tuplet,
            anchor_loss=anchor
        )

        return output_dict


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,torch.where(update_norm > 0,(g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)
                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])
                
def sparse_reconstruction_loss(x_true,x_pred,alpha=1.0,beta=1.0):
    id_true_nonzeros = (x_true!=0)
    #l_nonzero = torch.sqrt(((x_true[id_true_nonzeros] - x_pred[id_true_nonzeros])**2)+1e-16)
    #l_zero = torch.sqrt(((x_true[~id_true_nonzeros] - x_pred[~id_true_nonzeros])**2)+1e-16)
    l_nonzero = ((x_true[id_true_nonzeros] - x_pred[id_true_nonzeros])**2)
    l_zero = ((x_true[~id_true_nonzeros] - x_pred[~id_true_nonzeros])**2)
    l_nonzero  = l_nonzero.mean()
    l_zero  = l_zero.mean()

    recon_loss = (alpha   * l_nonzero) + (beta   * l_zero)

    return recon_loss
def reci(input_dict):
    return - input_dict['average_squared_distances'].max(dim=1)[0] 


class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random',device=0):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers
        

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim).to(device))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers 
            else:
                center = center 
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2) 

        return dist

class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, known_classes):
        super(RPLoss, self).__init__()
        self.weight_pl = 0.1
        self.temp = 1.0
        self.Dist = Dist(num_classes=known_classes, feat_dim=512, num_centers=known_classes, device=0)
        self.radius = 1

        self.radius = nn.Parameter(torch.Tensor(self.radius).to(0))
        self.radius.data.fill_(0)

    def forward(self, x, labels=None):
        dist = self.Dist(x)
        logits = F.softmax(dist, dim=1)
        if labels is None: return logits, 0
        loss = F.cross_entropy(dist / self.temp, labels)
        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius)
        loss = loss + self.weight_pl * loss_r

        return logits, loss, dist