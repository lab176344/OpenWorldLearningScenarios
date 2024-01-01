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
import torch.utils.data as data
from .utils import TransformTwice, GaussianBlur
from PIL import Image
import torchio
from einops import rearrange

class MyDataset(Dataset):
    def __init__(self, stuff_in, mat_path, mode='train',target_list=range(10),trans = None):
        self.stuff = stuff_in
        self.transforms_ = trans
        
        if(mode=='train'):
     
            data1  = torch.load(mat_path+'/Boundary/argoverse_Boundaryimage.pt')
            X_train = torch.stack(data1)
            X_train = self.correct_order_samples(X_train,[0,2,4,6,8,9,10,12,14,16])
            
            ag_data1  = torch.load(mat_path+'/Boundary/argoverse_BoundaryAugimage.pt')
            ag_X_train = torch.stack(ag_data1)
            ag_X_train = self.correct_order_samples(ag_X_train,[0,2,4,6,8,9,10,12,14,16])

           
            data_gt  = np.ones((X_train.shape[0]))
            data_gt = np.squeeze(data_gt)
            data_gt = torch.from_numpy(data_gt) 
    
            self.images = X_train
            self.images_aug = ag_X_train
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
            self.images_aug = test_set
            self.target = data_gt
        
              
    def correct_order_samples(self,data_in,stuff):
        data_out = torch.zeros((data_in.shape[0],1,len(stuff),data_in.shape[3],data_in.shape[4]))
        for j in range(data_in.shape[0]):
            for k in range(len(stuff)):
                data_out[j,:,k,:,:] = data_in[j,:,stuff[k],:,:]
        return data_out 

    def __getitem__(self, index):
        x = self.images[index]# bxcxfxhxw
        x_aug = self.images_aug[index]
        y = self.target[index]
        return (x,x_aug), y  
                
    def __len__(self):
        return len(self.images)
   
def get_collate(batch_transform=None):
    def mycollate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        if batch_transform is not None:
            collated = batch_transform(collated)
        return collated
    return mycollate

def generate_random_sequence(scenario1, scenario2, stuff, transform_,transform2_,rseed1,rseed2,test):
    """[Generates random sequences based on the OGs and the number of OGs chosen]

    Args:
        scenario ([array (L,W,T)]): [The scenario as stacked occupancy grids]
        stuff ([list]): [The correct order and the OGs to be chosen for the SSL]
        tranform_ ([type]): [transforems]

    Returns:
        [tuple]: [description]
    """

    trans_clip1 = []
    trans_clip2 = []

    random.seed(rseed1)
    torch.manual_seed(rseed1)
    trans_clip1 = transform_(scenario1)
    
  
    if not test:   
        random.seed(rseed2)
        torch.manual_seed(rseed2)
        trans_clip2 = transform2_(scenario2)
    else:
        random.seed(rseed2)
        torch.manual_seed(rseed2)
        trans_clip2 = transform_(scenario2)
        
    return trans_clip1,trans_clip2 

class DataLoader(object):
    def __init__(self,
                  dataset,
                  batch_size=1,
                  epoch_size=None,
                  num_workers=0,
                  shuffle=True,
                  trans = None,
                  trans2 = None,
                  test = False,
                  grid_chosen = 4,
                  sampler=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = trans
        self.transform2 = trans2
        self.test = test
        self.grid_chosen = grid_chosen
        self.epoch = 0
        self.sampler = sampler

    def get_iterator(self, epoch=0, batch=0):
        
        self.rand_seed1 = epoch 
        self.rand_seed2 = epoch+1


        # shuffle the grids and create a dataset with different combinations of 
        # temporal order and a corresponding label
        # EG (0,1,2) can be reshuffled as (1,0,2), (2,1,0), etc. 

        def _load_function(idx):
            idx = idx % len(self.dataset)
            (img1, img2), y = self.dataset[idx]
            rotated_imgs1,rotated_imgs2 = generate_random_sequence(img1, img2, self.grid_chosen, self.transform,self.transform2,self.rand_seed1,self.rand_seed2,self.test)
            return (rotated_imgs1,rotated_imgs2),y,idx


        def _collate_fun(batch):
            self.rand_seed1  += 3
            self.rand_seed2  += 10
            batch = default_collate(batch)
            assert(len(batch)==3)
            return batch       

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle,sampler=self.sampler)
        return data_loader

    def __call__(self, epoch=0,batch_size = 0):
        return self.get_iterator(epoch,batch_size)






