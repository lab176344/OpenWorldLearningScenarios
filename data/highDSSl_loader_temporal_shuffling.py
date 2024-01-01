from torch.utils.data.dataset import Dataset
import scipy.io as io
import torch
import tqdm
import torchvision.transforms as transforms
import numpy as np
import itertools
import random
from torch.utils.data.dataloader import default_collate
import torchnet as tnt
from random import randint
from PIL import Image
import math
from einops import rearrange

#TO DO Make this generic
class MyDataset(Dataset):
    def __init__(self, stuff_in, mat_path, mode='train',target_list=range(10),trans = None):
        self.stuff = stuff_in
        self.transforms_ = trans
        
        if(mode=='train'):
     
            data1  = torch.load(mat_path+'/Boundary/argoverse_Boundaryimage.pt')
            X_train = torch.stack(data1)
            X_train = self.correct_order_samples(X_train,[0,2,4,6,8,9,10,12,14,16])

            data_gt  = np.ones((X_train.shape[0]))
            data_gt = np.squeeze(data_gt)
            data_gt = torch.from_numpy(data_gt) 
    
            self.images = X_train
            self.target = data_gt

        elif(mode=='test'):
   
            # TO DO Update the
            data1 = torch.load(mat_path+'/Boundary/openTraffic_Boundary_image.pt')
            data1 = torch.stack(data1)
            
            SampleList = [1,10,20,30,40,50,60,70,80,90]
            SampleList[:] = [x-1 for x in SampleList]
            data1 = self.correct_order_samples(data1,SampleList)
            
            data1gt = torch.load(mat_path+'/Boundary/openTraffic_Boundarygt_image.pt') 
            data1gt = torch.stack(data1gt)
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XTrain1']
            data3 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XTrain2']             
            data4 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XTrain3']
            data5 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XTrain4']
            data6 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XTrain5']             
            data7 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XTrain6']              
            data8 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['XCalib'] 
            X_test = np.concatenate((data2,data3,data4,data5,data6,data7,data8))
            X_test = rearrange(X_test, 'b h w f->b 1 f h w')
            X_test = torch.from_numpy(X_test)
            X_test = torch.cat((X_test,data1),0)
            
            
            data_gt1 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['yTrain'] 
            data_gt2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res105.mat')['yCalib'] 
            data_gt  = np.concatenate((data_gt1,data_gt2))
            data_gt = torch.from_numpy(data_gt)
            data_gt = torch.cat((data_gt,data1gt),0)            
            data_gt = torch.squeeze(data_gt)
            
            
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            test_set = X_test[ind]
            data_gt = data_gt[ind]     
            self.images = test_set
            self.target = data_gt
        
              
    def correct_order_samples(self,data_in,stuff):
        data_out = torch.zeros((data_in.shape[0],1,len(stuff),data_in.shape[3],data_in.shape[4]))
        for j in range(data_in.shape[0]):
            for k in range(len(stuff)):
                data_out[j,:,k,:,:] = data_in[j,:,stuff[k],:,:]
        return data_out 

    def __getitem__(self, index):
        x = self.images[index] # bxcxfxhxw
        y = self.target[index]
        return (x, y)  
                
    def __len__(self):
        return len(self.images)
   


def get_shuffle_id(stuff):
    """
    Args:
        stuff ([list]): [original grid list]

    Returns:
        [list of lists]: [all possible combination of the list]
    """
    list_to_shuffle = []
    for L in range(0,len(stuff)+1):
        for subset in itertools.permutations(stuff,L):
            if(len(subset)>(len(stuff)-1)):
                list_to_shuffle.append(subset)
    return list_to_shuffle             

def generate_random_sequence(scenario, stuff, tranform_):
    """[Generates random sequences based on the OGs and the number of OGs chosen]

    Args:
        scenario ([array (L,W,T)]): [The scenario as stacked occupancy grids]
        stuff ([list]): [The correct order and the OGs to be chosen for the SSL]
        tranform_ ([type]): [transforems]

    Returns:
        [tuple]: [description]
    """
    list_to_shuffle = get_shuffle_id(stuff)
    trans_tuple = []

    for idx, list_check in enumerate(list_to_shuffle):
        # fix seed, apply the sample `random transformation` for all frames in the clip 
        seed = random.random()
        idxFilter = torch.LongTensor(list_check)
        trans_clip = scenario[:,idxFilter,:,:]
        trans_tuple.append(trans_clip)
    return trans_tuple 

class DataLoader(object):
    def __init__(self,
                  dataset,
                  batch_size=1,
                  epoch_size=None,
                  num_workers=0,
                  shuffle=True,
                  trans = None,
                  grid_chosen = [0,3,6,9],
                  sampler=None,
                  num_gpus=1):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = trans
        self.grid_chosen = grid_chosen
        self.samples = sampler
        self.num_gpus = num_gpus    

    def get_iterator(self, epoch,gpu_idx):
        
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)

        # shuffle the grids and create a dataset with different combinations of 
        # temporal order and a corresponding label
        # EG (0,1,2) can be reshuffled as (1,0,2), (2,1,0), etc. 

        def _load_function(idx):
            idx = idx % len(self.dataset)
            img, _ = self.dataset[idx]
            no_labels = math.factorial(len(self.grid_chosen))
            rotated_imgs = generate_random_sequence(img, self.grid_chosen, self.transform)
            rotation_labels = torch.LongTensor(list(range(no_labels)))
            return torch.stack(rotated_imgs, dim=0), rotation_labels


        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch)==2)
            batch_size, shuffle, channels, height, width, depth  = batch[0].size()
            batch[0] = batch[0].view([batch_size*shuffle, channels, height, width, depth])
            batch[1] = batch[1].view([batch_size*shuffle])
            return batch       

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            tnt_dataset,
            num_replicas=self.num_gpus,
            shuffle=self.shuffle,
            rank=gpu_idx)
        sampler.set_epoch(epoch)
        
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun,num_workers=self.num_workers,
            sampler=sampler)
        return data_loader

    def __call__(self, epoch=0,rank=0):
        return self.get_iterator(epoch,rank)

    def __len__(self):
        return self.epoch_size / self.batch_size
