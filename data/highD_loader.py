from torch.utils.data.dataset import Dataset
from torch.utils.data import  Subset

import scipy.io as io
import torch
import tqdm
import torchvision.transforms as transforms
import numpy as np
import itertools
import random
from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
from .utils import TransformTwice, GaussianBlur
import torchio
from einops import rearrange
from sklearn.model_selection import train_test_split
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    )
class MyDataset(Dataset):
    def __init__(self, stuff_in, mat_path, mode='train', target_list=range(10),
                 transform=None, transform_3d=None, aug=None, sizeMatch=False, matchSize=None):
        self.transform = transform
        self.stuff = stuff_in
        self.aug = aug
    
        # Load the opentraffic dataset for the OSR and clustering in singleshot mode
        if mode is not 'unknownTrain' and mode is not 'unknownTest':
            dataopenTraffic = torch.load(mat_path+'/Boundary/openTraffic_Boundary_image.pt')
            dataopenTraffic = torch.stack(dataopenTraffic)
            dataopenTrafficSim = torch.load(mat_path+'/Boundary/openTrafficSim_Boundary_image.pt')
            dataopenTrafficSim = torch.stack(dataopenTrafficSim)
            
            SampleList = [1,10,20,30,40,50,60,70,80,90]
            SampleList[:] = [x-1 for x in SampleList]
            dataopenTraffic = self.correct_order_samples(dataopenTraffic,SampleList)
            dataopenTrafficSim = self.correct_order_samples(dataopenTrafficSim,SampleList)
            
            dataopenTrafficGt = torch.load(mat_path+'/Boundary/openTraffic_Boundarygt_image.pt') 
            dataopenTrafficSimGt = torch.load(mat_path+'/Boundary/openTrafficSim_Boundarygt_image.pt')
            
            dataopenTrafficGt = torch.stack(dataopenTrafficGt)
            dataopenTrafficGt = torch.squeeze(dataopenTrafficGt)
            
            dataopenTrafficSimGt = torch.stack(dataopenTrafficSimGt)
            dataopenTrafficSimGt = torch.squeeze(dataopenTrafficSimGt)
            
            dataopenTraffic = torch.cat((dataopenTraffic,dataopenTrafficSim),0)
            dataopenTrafficGt = torch.cat((dataopenTrafficGt,dataopenTrafficSimGt))
            dataopenTrafficGt = dataopenTrafficGt  - 2         
            XTrainOT,yTrainOT,XTestOT,yTestOT,XCalibOT,yCalibOT = self.testValTrainSplit(dataopenTraffic,dataopenTrafficGt)
        # *******************************************************************************************************************
        # EQUAL data splitting
        # *******************************************************************************************************************
        # OSR training   
        if(mode=='train'):
            # TO DO Update the   
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain1']
            data3 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain2']             
            data4 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain3']
            data5 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain4']
            data6 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain5']             
             
            XTrain = np.concatenate((data2,data3,data4,data5,data6))
            XTrain = rearrange(XTrain, 'b h w f->b 1 f h w')
            XTrain = torch.from_numpy(XTrain)
            XTrain = torch.cat((XTrain,XTrainOT),0)
                   
            data_gt = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yTrain']
            data_gt = torch.from_numpy(data_gt)
            data_gt = torch.squeeze(data_gt) 
            data_gt = torch.cat((data_gt,yTrainOT),0)          

            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            train_set = XTrain[ind]
            data_gt = data_gt[ind]    
            if train_set.shape[0]>sizeMatch:
                perm =  torch.randperm(train_set.shape[0])
                idx = perm[:matchSize]

                train_set,data_gt = train_set[idx,:,:,:,:],data_gt[idx] 
            self.images = train_set
            self.target = data_gt
        # OSR Testing
        elif(mode=='test'):
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTest']
            data2 = rearrange(data2, 'b h w f->b 1 f h w')
            data2 = torch.from_numpy(data2)
            XTest = torch.cat((data2,XTestOT),0)
            data_gt = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yTest'] 
            data_gt = torch.from_numpy(data_gt)
            data_gt = torch.squeeze(data_gt)
            data_gt = torch.cat((data_gt,yTestOT),0)            
                    
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            test_set = XTest[ind]
            data_gt = data_gt[ind]     
            self.images = test_set
            self.target = data_gt
        # OSR calibration
        elif(mode=='calib'):
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XCalib']
            data2 = rearrange(data2, 'b h w f->b 1 f h w')
            data2 = torch.from_numpy(data2)
            X_Calib = torch.cat((data2,XCalibOT),0)
            data_gt = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yCalib'] 
            data_gt = torch.from_numpy(data_gt)
            data_gt = torch.squeeze(data_gt)

            data_gt = torch.cat((data_gt,yCalibOT),0)            
                    
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            test_set = X_Calib[ind]
            data_gt = data_gt[ind]     
            self.images = test_set
            self.target = data_gt    
            
        
        # Buffer mode training dataset (seed is common for testing and training split)   
        elif(mode=='unknownTrain'): 
            if type(mat_path) is list:
                data = []
                data_gt = []
                data_gt_osr = []
                for mattemp in mat_path:
                    data_temp = (io.loadmat(mattemp)['unknown_input_collect'])
                    data_gt_temp = io.loadmat(mattemp)['unknown_Tlabels_collect']
                    data_gt_osr_temp = io.loadmat(mattemp)['unknown_labels_collect']
                    data.append(data_temp)
                    data_gt.append(data_gt_temp)
                    data_gt_osr.append(data_gt_osr_temp)
                data = np.concatenate(data, axis=0)
                data_gt = np.concatenate(data_gt, axis=1).squeeze(0)
                data_gt_osr = np.concatenate(data_gt_osr, axis=1).squeeze(0)
            else:         
                data = (io.loadmat(mat_path)['unknown_input_collect'])
                data_gt = io.loadmat(mat_path)['unknown_Tlabels_collect']
                data_gt = np.squeeze(data_gt)
                data_gt_osr = io.loadmat(mat_path)['unknown_labels_collect']
            XTrain,XTest,yTrain,yTest = self.testTrainSplit(data,data_gt)
            train_set = self.change_data_size(XTrain)
            data_gt = yTrain.tolist()
            data_gt = np.array(data_gt)
            data_gt = np.squeeze(data_gt)
            self.images = train_set                  
            self.target = torch.from_numpy(data_gt) 
            
        # Buffer mode testing dataset (seed is common for testing and training split)  
        elif(mode=='unknownTest'):          
            if type(mat_path) is list:
                data = []
                data_gt = []
                data_gt_osr = []
                for mattemp in mat_path:
                    data_temp = (io.loadmat(mattemp)['unknown_input_collect'])
                    data_gt_temp = io.loadmat(mattemp)['unknown_Tlabels_collect']
                    data_gt_osr_temp = io.loadmat(mattemp)['unknown_labels_collect']
                    data.append(data_temp)
                    data_gt.append(data_gt_temp)
                    data_gt_osr.append(data_gt_osr_temp)
                data = np.concatenate(data, axis=0)
                data_gt = np.concatenate(data_gt, axis=1).squeeze(0)
                data_gt_osr = np.concatenate(data_gt_osr, axis=1).squeeze(0)
            else:         
                data = (io.loadmat(mat_path)['unknown_input_collect'])
                data_gt = io.loadmat(mat_path)['unknown_Tlabels_collect']
                data_gt = np.squeeze(data_gt)
                data_gt_osr = io.loadmat(mat_path)['unknown_labels_collect']
            XTrain,XTest,yTrain,yTest = self.testTrainSplit(data,data_gt)
            test_set = self.change_data_size(XTest)
            data_gt = yTest.tolist()
            data_gt = np.array(data_gt)
            data_gt = np.squeeze(data_gt)
            self.images = test_set                  
            self.target = torch.from_numpy(data_gt)         
        if mode is 'all':
            #  Train 
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain1']
            data3 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain2']             
            data4 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain3']
            data5 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain4']
            data6 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain5']                           
            XTrain = np.concatenate((data2,data3,data4,data5,data6))
            XTrain = rearrange(XTrain, 'b h w f->b 1 f h w')
            XTrain = torch.from_numpy(XTrain)
            XTrain = torch.cat((XTrain,XTrainOT),0)
            
            
            data_gt = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yTrain']
            data_gt = torch.from_numpy(data_gt)
            data_gt = torch.squeeze(data_gt) 

            data_gt = torch.cat((data_gt,yTrainOT),0)     
            #Test       
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTest']
            data2 = rearrange(data2, 'b h w f->b 1 f h w')
            data2 = torch.from_numpy(data2)
            XTrain = torch.cat((XTrain,data2,XTestOT),0)
            data_gtTest = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yTest'] 
            data_gtTest = torch.from_numpy(data_gtTest)
            data_gtTest = torch.squeeze(data_gtTest)
            data_gt = torch.cat((data_gt,data_gtTest,yTestOT),0)            
                    
            # Calib
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XCalib']
            data2 = rearrange(data2, 'b h w f->b 1 f h w')
            data2 = torch.from_numpy(data2)
            XTrain = torch.cat((XTrain,data2,XCalibOT),0)
            data_gtCalib = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yCalib'] 
            data_gtCalib = torch.from_numpy(data_gtCalib)
            data_gtCalib = torch.squeeze(data_gtCalib)

            data_gt = torch.cat((data_gt,data_gtCalib,yCalibOT),0)               
            
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            train_set = XTrain[ind]
            data_gt = data_gt[ind]
            self.images = train_set
            self.target = data_gt         
        # ****************************************************************************************************************** 
        # Unequal data split for training and testing       
        # Ignore the split from matlab and split unequally in splitUnequal mode or mode all in clustering setup  or if it is explicity asked for all
        # ******************************************************************************************************************
        if mode is 'splitUnequal' or mode is 'trainUE' or mode is 'testUE' or mode is 'calibUE':
            #  Train 
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain1']
            data3 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain2']             
            data4 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain3']
            data5 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain4']
            data6 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTrain5']                          
            XTrain = np.concatenate((data2,data3,data4,data5,data6))
            XTrain = rearrange(XTrain, 'b h w f->b 1 f h w')
            XTrain = torch.from_numpy(XTrain)
            XTrain = torch.cat((XTrain,XTrainOT),0)
            
            
            data_gt = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yTrain']
            data_gt = torch.from_numpy(data_gt)
            data_gt = torch.squeeze(data_gt) 

            data_gt = torch.cat((data_gt,yTrainOT),0)     
            #Test       
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XTest']
            data2 = rearrange(data2, 'b h w f->b 1 f h w')
            data2 = torch.from_numpy(data2)
            XTrain = torch.cat((XTrain,data2,XTestOT),0)
            data_gtTest = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yTest'] 
            data_gtTest = torch.from_numpy(data_gtTest)
            data_gtTest = torch.squeeze(data_gtTest)
            data_gt = torch.cat((data_gt,data_gtTest,yTestOT),0)            
                    
            # Calib
            data2 = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['XCalib']
            data2 = rearrange(data2, 'b h w f->b 1 f h w')
            data2 = torch.from_numpy(data2)
            XTrain = torch.cat((XTrain,data2,XCalibOT),0)
            data_gtCalib = io.loadmat(mat_path+'/Boundary/Dataset_check_120_Boundary_Res10513classes.mat')['yCalib'] 
            data_gtCalib = torch.from_numpy(data_gtCalib)
            data_gtCalib = torch.squeeze(data_gtCalib)

            data_gt = torch.cat((data_gt,data_gtCalib,yCalibOT),0)               
            
            ind = [i for i in range(len(data_gt)) if data_gt[i] in target_list]
            train_set = XTrain[ind]
            data_gt = data_gt[ind]
            data = train_set.numpy()
            label = data_gt.numpy()
            XTrainSplitUE,XTestSplitUE,yTrainSplitUE,yTestSplitUE = train_test_split(data,label,test_size=0.5,random_state=42,stratify=label)
            XTestSplitUE,XCalibSplitUE,yTestSplitUE,yCalibSplitUE = train_test_split(XTestSplitUE,yTestSplitUE,test_size=0.5,random_state=42)
        if mode=='trainUE': #train unequal 
            self.images = torch.from_numpy(XTrainSplitUE)
            self.target = torch.from_numpy(yTrainSplitUE)
        elif mode=='testUE':#test unequal    
            self.images = torch.from_numpy(XTestSplitUE)
            self.target = torch.from_numpy(yTestSplitUE)
        elif mode=='calibUE':#calib unequal
            self.images = torch.from_numpy(XCalibSplitUE)
            self.target = torch.from_numpy(yCalibSplitUE)
                        
    def correct_order_samples(self,data_in,stuff):
        data_out = torch.zeros((data_in.shape[0],1,len(stuff),data_in.shape[3],data_in.shape[4]))
        for j in range(data_in.shape[0]):
            for k in range(len(stuff)):
                data_out[j,:,k,:,:] = data_in[j,:,stuff[k],:,:]
        return data_out
    
    def change_data_size(self,data):
        transform = transforms.Compose([
                transforms.Resize(120),
                ])
        data_120 = torch.zeros((data.shape[0],1,10,120,120))
        id = [0,3,6,9]
        for i in range(data.shape[0]):
            data_120[i,:,id,:,:] = (data[i,:,:,:,:])
        return data_120
    
    def testTrainSplit(self,data,label):
        XTrain,XTest,yTrain,yTest = train_test_split(data,label,test_size=0.33,random_state=42)
        XTrain = torch.from_numpy(XTrain)
        XTest = torch.from_numpy(XTest)
        return XTrain,XTest,yTrain,yTest
    
    
    def testValTrainSplit(self,data,label):
        data = data.numpy()
        label = label.numpy()
        XTrain,XTest,yTrain,yTest = train_test_split(data,label,test_size=0.33,random_state=42,stratify=label)
        XTest,XCalib,yTest,yCalib = train_test_split(XTest,yTest,test_size=0.4,random_state=42)
        XTrain = torch.from_numpy(XTrain)
        XTest = torch.from_numpy(XTest)
        XCalib = torch.from_numpy(XCalib)
        yTrain = torch.from_numpy(yTrain)
        yTest = torch.from_numpy(yTest)
        yCalib = torch.from_numpy(yCalib)
        XTrain,yTrain = self.selectTestTrainVal(XTrain,yTrain,2880)
        XTest,yTest = self.selectTestTrainVal(XTrain,yTrain,810)
        XCalib,yCalib = self.selectTestTrainVal(XTrain,yTrain,450)
        
        return XTrain,yTrain,XTest,yTest,XCalib,yCalib
    
    def selectTestTrainVal(self,data,label,sizeRequired):
        uniqueClasses = np.unique(label)
        balancedClasses = int(sizeRequired/len(uniqueClasses))
        dataOut = torch.zeros((balancedClasses*len(uniqueClasses),data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        labelOut = torch.zeros((balancedClasses*len(uniqueClasses),))
        classCount = torch.zeros((len(uniqueClasses),))
        dataFill = 0
        for idx,(x,y) in enumerate(zip(data,label)):
            if classCount[np.where(uniqueClasses==y.numpy())]<balancedClasses:
                classCount[np.where(uniqueClasses==y.numpy())]+=1
                dataOut[dataFill,:,:,:,:] = x
                labelOut[dataFill] = y
                dataFill+=1
            if torch.sum(classCount)==balancedClasses*len(uniqueClasses):
                break
                
        return dataOut,labelOut
                    
        
             

    def __getitem__(self, index):
        x = self.images[index]
        y = self.target[index]
        if self.aug =='twice':
            # fix seed, apply the sample `random transformation` for all frames in the clip 

            trans_clip1 = self.transform(x)

            trans_clip2 = self.transform(x)
   
            return (trans_clip1,trans_clip2), torch.tensor(int(y)), index  
                
        else:
            if self.transform:
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                trans_clip = self.transform(x)           
            else:
                trans_clip = x
    
            return trans_clip, torch.tensor(int(y)), index  
 
    def __len__(self):
        return len(self.images)
    

    
def ScenarioLoaderMix(root, batch_size, split='train',  grid_chosen=[0,3,6,9], num_workers=0, aug=None, shuffle=True, labeled_list=range(10), unlabeled_list=range(10, 16), new_labels=None):


    if aug==None:
        transform = transforms.Compose([
        UniformTemporalSubsample(4),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            UniformTemporalSubsample(4),
+            transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40))],p=0.5),
            transforms.GaussianBlur(1),

        ])
    elif aug=='twice':
        transform = transforms.Compose([
            UniformTemporalSubsample(4),
            transforms.RandomApply(transforms=[
                transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40))],p=0.5),
            transforms.GaussianBlur(1),
        ])
        
    if split=='testplusunknown':
        split_labelled = 'test'
        split_unlabeled = 'all'
    elif split == 'testplusunknownUE':
        split_labelled = 'testUE'
        split_unlabeled =  'all'
    elif split=='unknownTrain':
        split_labelled = 'train'
        split_unlabeled = 'unknownTrain'
    else:
        split_labelled = split
        split_unlabeled = split 
         

    transforms_3d = transforms.Compose([torchio.transforms.RandomNoise(std=(0,0.0000000000000001))]) 
    dataset_labeled = MyDataset(stuff_in=grid_chosen,mat_path=root, transform=transform, transform_3d= transforms_3d, mode=split_labelled, target_list=labeled_list, aug=aug)
    print('Labelled Data shape', dataset_labeled.images.shape)
    dataset_unlabeled = MyDataset(stuff_in=grid_chosen,mat_path=root, transform=transform, transform_3d= transforms_3d, mode=split_unlabeled, target_list=unlabeled_list, aug=aug)
    print('Unlabelled Data shape', dataset_unlabeled.images.shape)

    dataset_labeled.target = torch.cat((dataset_labeled.target,dataset_unlabeled.target))
    dataset_labeled.images = torch.cat((dataset_labeled.images,dataset_unlabeled.images),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    if split=='train':
        print('All Data shape', dataset_labeled.images.shape)
        print('Unabelled Data shape', dataset_unlabeled.images.shape)
       
    else:
        print('All Data shape', dataset_labeled.images.shape)
        print('Unabelled Data shape', dataset_unlabeled.images.shape)

    return loader

def ScenarioLoaderBuffer(root, unknown_root, batch_size, split='train',  grid_chosen=[0,3,6,9], num_workers=0, aug=None, shuffle=True, labeled_list=range(10),sizeMatch=False):


    if aug==None:
        transform = transforms.Compose([
        UniformTemporalSubsample(4),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            UniformTemporalSubsample(4),
            transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40))],p=0.5),
            transforms.GaussianBlur(1),
        ])
    elif aug=='twice':
        transform = transforms.Compose([
            UniformTemporalSubsample(4),
            transforms.RandomApply(transforms=[
                transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40))],p=0.5),
            transforms.GaussianBlur(1),

        ])
        
    if split=='testplusunknown':
        split_labelled = 'test'
        split_unlabeled = 'all'
    elif split == 'testplusunknownUE':
        split_labelled = 'testUE'
        split_unlabeled =  'all'
    elif split=='unknownTrain':
        split_labelled = 'train'
        split_unlabeled = 'unknownTrain'
    else:
        split_labelled = split
        split_unlabeled = split  
        
    transforms_3d = transforms.Compose([torchio.transforms.RandomNoise(std=(0,0.0000000000000001))])
    dataset_unlabeled = MyDataset(stuff_in=grid_chosen,mat_path=unknown_root, transform=transform, transform_3d= transforms_3d, mode=split_unlabeled, aug=aug)
    unlabelledDataSetShape = dataset_unlabeled.images.shape[0]
    dataset_labeled = MyDataset(stuff_in=grid_chosen,mat_path=root, transform=transform, transform_3d= transforms_3d, mode=split_labelled, target_list=labeled_list,
                                aug=aug,sizeMatch=sizeMatch,matchSize=unlabelledDataSetShape)
    print('Labelled Data shape', dataset_labeled.images.shape)
    # generate subset based on indices

    dataset_labeled.target = torch.cat((dataset_labeled.target,dataset_unlabeled.target))
    dataset_labeled.images = torch.cat((dataset_labeled.images,dataset_unlabeled.images),0)

    
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    if split=='train':
        print('All Data shape', dataset_labeled.images.shape)
        print('Unabelled Data shape', dataset_unlabeled.images.shape)
       
    else:
        print('All Data shape', dataset_labeled.images.shape)
        print('Unabelled Data shape', dataset_unlabeled.images.shape)

    return loader


def ScenarioLoader(root,batch_size,unknown_root=None, split='train',aug=None, grid_chosen=[0,3,6,9], num_workers=0, shuffle=True, target_list=range(4)):

    if aug==None:
        transform = transforms.Compose([
        UniformTemporalSubsample(4),
        ])
    elif aug=='once':
        transform = transforms.Compose([
            UniformTemporalSubsample(4),
            transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40))],p=0.5),
            transforms.GaussianBlur(1),
     
        ])
    elif aug=='twice':
        transform = transforms.Compose([
            UniformTemporalSubsample(4),
            transforms.RandomApply(transforms=[
                transforms.RandomRotation(degrees=(-10, 10), fill=(0,),center=(20,40))],p=0.5),
            transforms.GaussianBlur(1),

        ])
    transforms_3d = transforms.Compose([torchio.transforms.RandomNoise(std=(0,0.0000000000000001))]) 

    if split=='testBuffer':
        splitLabelled = 'test'
        splitUnlabeled = 'unknownTest'
        
        dataset_unlabeled = MyDataset(stuff_in=grid_chosen,mat_path=unknown_root, transform=transform, transform_3d= transforms_3d, mode=splitUnlabeled, aug=aug)
        unlabelledDataSetShape = dataset_unlabeled.images.shape[0]
        dataset_labeled = MyDataset(stuff_in=grid_chosen,mat_path=root, transform=transform, transform_3d= transforms_3d, mode=splitLabelled, target_list=target_list,
                                    aug=aug)
        dataset_labeled.target = torch.cat((dataset_labeled.target,dataset_unlabeled.target))
        dataset_labeled.images = torch.cat((dataset_labeled.images,dataset_unlabeled.images),0)
        dataset = dataset_labeled
        
    else:
        split_check = split
        dataset = MyDataset(stuff_in=grid_chosen, mat_path=root, transform=transform,transform_3d= transforms_3d, mode=split_check, target_list=target_list, aug=aug)
        datasetLength = dataset.images.shape[0]
    
    loader = data.DataLoader(dataset, batch_size=batch_size,  shuffle=shuffle, num_workers=num_workers)

    print('{} Data shape {}'.format(split, dataset.images.shape))
    return loader