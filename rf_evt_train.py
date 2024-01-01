import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import torch
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
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import weibull_min
import matlab.engine
import matlab
from models.timesformer_supervised import TimeSformer
import pickle as cPickle
from utils.util import cluster_acc
import json 
import torch
from einops import rearrange, repeat


    
def train_RF(features, lables, args):
    clf = RandomForestClassifier(n_estimators=args.num_trees,n_jobs=-1)
    clf.fit(features, lables)
    return clf

def test_RF(features, labels, clf, args):
    pred_labels = clf.predict(features)
    threshold = []
    for n_class in range((args.num_labeled_classes)):
        pred_class_Labels = [i for i,j in zip(pred_labels,labels) if j==n_class]
        true_class_Labels = [j for j in labels if j==n_class]
        acc = cluster_acc(np.array(true_class_Labels).astype(int), np.array(pred_class_Labels).astype(int))
        # accuracy or n tail fixed
        threshold.append(args.n_tail)
        #threshold.append(acc)
        args.classWiseThres = threshold
        print('Accuracy for class {} is {}'.format(n_class,acc))
    return cluster_acc(labels.astype(int), pred_labels.astype(int)) 

def train_evt(features, labels, clf, args):
    param_vec = []
    eng = matlab.engine.start_matlab()
    for n_class in range((args.num_labeled_classes)):
        print('Training EVT for class {}'.format(n_class))
        c_trees = [(args.num_trees*clf.predict_proba(i.reshape(1, -1))) for i,j in zip(features,labels) if j==n_class]
        c1_max = [np.max(i) for i in c_trees]
        c1_max = np.sort(np.array(c1_max))
        threshold_number = np.sum(c1_max<=(args.classWiseThres[n_class]*args.num_trees))
        c_tail = matlab.double(c1_max[:threshold_number].tolist())
        print(len(c1_max[:threshold_number].tolist()))
        if len(c1_max[:threshold_number].tolist())==0:
            print('!!!! Class {} has no tail values !!!!!'.format(n_class))
        parmhat = eng.wblfit(c_tail) 
        param = np.array(parmhat)
        param_vec.append(param.tolist())
        print('Finished EVT for class {}'.format(n_class))

    return param_vec   

def extract_features(model,data_loader,args):
    model.eval()
    features_model = np.zeros((len(data_loader.dataset),512))
    label_array = np.zeros((len(data_loader.dataset),),dtype=int)
    with torch.no_grad():
        for _, (x, label, idx) in enumerate(tqdm(data_loader)):
            x, label = x.to(device), label.to(device)
            if args.model_use=='Times':
                x = rearrange(x, 'b c f h w  -> b f c h w ')
            _,  features = model(x)
            features_model[idx,:] = features.cpu().numpy()
            label_array[idx,] = label.cpu().numpy()
    return features_model, label_array
      

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='rf_evt',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_labeled_classes', default=6, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--model_name', type=str, default='ssl_labelled')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_use', type=str, default='ResNet')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--UMAP', type=bool, default=False)
    parser.add_argument('--img_width', type=int, default=120)
    parser.add_argument('--img_height', type=int, default=120)
    parser.add_argument('--img_depth', type=int, default=4)
    parser.add_argument('--num_trees', type=int, default=250)
    parser.add_argument('--n_tail', type=float, default=0.9) # changethis later
    parser.add_argument('--exp_root', type=str, default='./data/experiments/supervised_learning_scenarios/')
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--SSL', type=str, default='Barlow')
    parser.add_argument('--split_strategy', type=str, default='equal') # 'equal' or 'unequal'

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, args.model_use, str(args.model_depth))
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.model_use == 'ResNet':
        args.model_name = args.model_name + '_resnet_' + str(args.model_depth) + '_' +str(args.SSL)+ '_' + str(args.num_labeled_classes)
    else:
        args.model_name = args.model_name + '_times_' +str(args.SSL)+ '_' + str(args.num_labeled_classes)
        
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name)
    args.rf_save_dir = model_dir+'/'+'{}'.format(args.model_name) +'_RF_{}'.format(args.num_trees)+ '_' + str(args.num_labeled_classes) + '/'
    args.evt_save_dir = model_dir+'/'+'{}'.format(args.model_name) +'_EVT_{}'.format(args.num_trees)+ '_' + str(args.num_labeled_classes) + '/'
    if not os.path.exists(args.rf_save_dir):
        os.makedirs(args.rf_save_dir)
    if not os.path.exists(args.evt_save_dir):
        os.makedirs(args.evt_save_dir)
    # Choose the model
    if args.model_use == 'ResNet':    
        model = generate_model(args.model_depth, args.num_labeled_classes) 
        model = model.to(device)
        state_dict = torch.load(args.model_dir)
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

        state_dict = torch.load(args.model_dir)
    print("model loaded from {}.".format(args.model_dir))
    model.load_state_dict(state_dict, strict=False)

    if args.split_strategy == 'equal':
        labeled_train_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', shuffle=True, aug='once', target_list = range(args.num_labeled_classes))
        labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', shuffle=False,aug=None, target_list = range(args.num_labeled_classes))
        labeled_calib_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='calib', shuffle=True,aug='once', target_list = range(args.num_labeled_classes))
    elif args.split_strategy == 'unequal':
        labeled_train_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='trainUE', shuffle=True, aug='once', target_list = range(args.num_labeled_classes))
        labeled_eval_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='testUE', shuffle=False,aug=None, target_list = range(args.num_labeled_classes))
        labeled_calib_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='calibUE', shuffle=True,aug='once', target_list = range(args.num_labeled_classes))        

    train_features, train_labels = extract_features(model,labeled_train_loader,args)
    test_features, test_labels = extract_features(model,labeled_eval_loader,args)
    calib_features, calib_labels = extract_features(model,labeled_calib_loader,args)

    rf = train_RF(calib_features, calib_labels, args)
    acc = test_RF(test_features, test_labels, rf, args)
    print('---------------------------------------------------')
    print('RF Trained')
    print('---------------------------------------------------')

    print('Test acc {:.4f}'.format(acc))
    rf_acc = {'RF_Acc':acc}

    args.rf_save_dir_acc = args.rf_save_dir + '\\' + 'RF_acc.json'   
    with open(args.rf_save_dir_acc, 'w') as f:
        json.dump(rf_acc, f)  
         
    args.rf_save_dir = args.rf_save_dir + '\\' + 'RF.pkl'   
    with open(args.rf_save_dir, 'wb') as f:
        cPickle.dump(rf, f)
    
    param_vec = train_evt(test_features, test_labels, rf, args)
    print('---------------------------------------------------')
    print('EVT Trained')
    print('---------------------------------------------------')
    evt_data = {'EVT_Param':param_vec}
    args.evt_save_dir = args.evt_save_dir + '\EVT_model_'+ args.model_name + '.json'
    with open(args.evt_save_dir, 'w') as outfile:
        json.dump(evt_data, outfile)
      
    print('---------------------------------------------------')
    print('End')
    print('---------------------------------------------------')
