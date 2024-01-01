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
from data.highD_loader import MyDataset,ScenarioLoaderMix,ScenarioLoader
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import weibull_min
import matlab.engine
import matlab
from models.timesformer_supervised import TimeSformer
import pickle as cPickle
from utils.util import cluster_acc
import json 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from scipy.special import softmax
import random
import scipy.io as io
from einops import rearrange, reduce, repeat
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models.timesformer_finetune import TimeSformer as TimeSformerCluser
from models.resnet3d_finetune import generate_model as generate_model_cluster
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.lof import LOF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import weibull_min
import matlab.engine
import matlab
from models.autoencoder_model_clustering import AE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from scipy.io import loadmat
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
def train_evt(features, labels, clf, uniqueClusterLabelsEVT):
    param_vec = []
    num_trees = 250
    eng = matlab.engine.start_matlab()
    for n_class in uniqueClusterLabelsEVT:
        print('Training EVT for class {}'.format(n_class))
        c_trees = [(num_trees*clf.predict_proba(i.reshape(1, -1))) for i,j in zip(features,labels) if j==n_class]
        c1_max = [np.max(i) for i in c_trees]
        c1_max = np.sort(np.array(c1_max))
        if len(c1_max) > 0:
            threshold_number = np.sum(c1_max<=(240))
            checkThreshold = 240
            while threshold_number<=1:
                checkThreshold += 1
                threshold_number = np.sum(c1_max<=(checkThreshold))
            c_tail = matlab.double(c1_max[:threshold_number].tolist())
            print(len(c1_max[:threshold_number].tolist()))
            if len(c1_max[:threshold_number].tolist())==0:
                print('!!!! Class {} has no tail values !!!!!'.format(n_class))
            parmhat = eng.wblfit(c_tail) 
            param = np.array(parmhat)
            param_vec.append(param.tolist())
        else:
            param_vec.append([0.0,0.0])
        print('Finished EVT for class {}'.format(n_class))

    return param_vec 
    
def train_RF(features, lables, args):
    clf = RandomForestClassifier(n_estimators=args.num_trees,n_jobs=-1)
    clf.fit(features, lables)
    return clf               
def plot_umap(data, label,epoch,args):
    U = umap.UMAP(n_components = 2)
    print('Shape_Extracted', data.shape)
    embedding2 = U.fit_transform(data,)
    label = label.astype(int)
    classesList = list(np.unique(label))
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
    
def osr_single_scenario(model,  rf_classifier, EVT_param, input_video, mat_eng, args,cluster_model=None, outlier_model=None,
                        clustered_classes=None,cluster_gTruth=None,param_vec_clustered=None):
    model.eval()
    with torch.no_grad():
        input_video = input_video.to(device)
        if args.model_use=='Times':
            input_video = rearrange(input_video, 'b c f h w  -> b f c h w ') 
        output, features = model(input_video)
        if cluster_model is not None:
            #_,cluster_features = model(input_video) 
            if args.model_use=='AE':
                _ , cluster_features, _ = cluster_model(input_video) 

            else:
                _ , _, cluster_features = cluster_model(input_video) 
            clusterId = outlier_model.predict(cluster_features.cpu().numpy())
            clusterId_prob = outlier_model.predict_proba(cluster_features.cpu().numpy())
        rf_class_pred = rf_classifier.predict(features.cpu().numpy())
        rf_class_prob = rf_classifier.predict_proba(features.cpu().numpy())
        preds=np.zeros((rf_class_pred.shape[0],),dtype=int)
        actual_preds=np.zeros((rf_class_pred.shape[0],),dtype=int)
    
    if args.OSR_method == 'EVT':
        for idx, n_class in enumerate(rf_class_pred):
            rf_tree_max = np.max(rf_class_prob[idx])*args.num_trees   
            paramhat = EVT_param[n_class]
            param_1 = matlab.double([paramhat[0][0]])
            param_2 = matlab.double([paramhat[0][1]])
            cnf_tree = matlab.double([rf_tree_max])          
            ConfFilterEVT = mat_eng.wblcdf(cnf_tree,param_1,param_2,0)
            # distrbution based filter
            #args.osr_filter = mat_eng.wblstat(param_1,param_2)
            #args.osr_filter =  args.osr_filter/args.num_trees
            
            # clusterId = 1 is outlier and 0 is inlier
            unknownDataBase = False
            if ConfFilterEVT<args.osr_filter:
                unknownDataBase = True
                if cluster_model is not None:
                    preds[idx,] = clustered_classes[-1]+1 # Unknown
                else:
                    preds[idx,] = args.num_labeled_classes # Unknown
            else:
                preds[idx,] = rf_class_pred[idx] # Known
                actual_preds[idx,] =  rf_class_pred[idx]
                
            #if cluster_model is not None and clusterId[idx]==0 and unknownDataBase: 
            #if cluster_gTruth[idx] == 3 or cluster_gTruth[idx] == 4:
            if cluster_model is not None:
                dataPointProb = clusterId_prob[idx]
                idclassEvTunknown = clusterId[idx]
                paramVecEVT = param_vec_clustered[idclassEvTunknown]
                param_1EVT = matlab.double([paramhat[0][0]])
                param_2EVT = matlab.double([paramhat[0][1]])
                rf_tree_maxEVT = np.max(dataPointProb)*args.num_trees
                cnf_treeEVT = matlab.double([rf_tree_maxEVT])          
                ConfFilterEVT = mat_eng.wblcdf(cnf_treeEVT,param_1EVT,param_2EVT,0)
            if cluster_model is not None and ConfFilterEVT>=0.5 and unknownDataBase:
                print(cluster_gTruth[idx])
                preds[idx,] = clustered_classes[0] # if from the clustered class assign the clustered_class id 0, eg [5,6] be the clustered class 5 will be assigned  
        return  preds
    
    elif args.OSR_method == 'SoftmaxArg':
        output = output.cpu().numpy()
        output = softmax(output, axis=1)
        class_id = np.argmax(output,axis=1)
        return class_id
    
    elif args.OSR_method == 'RFArg':
        return rf_class_pred
    
    elif args.OSR_method == 'SoftThresh':
        output = output.cpu().numpy()
        output = softmax(output, axis=1)
        class_id = np.argmax(output,axis=1)
        class_score = np.max(output,axis=1)

        for idx,(clss_idx,score) in enumerate(zip(class_id,class_score)):
            paramhat = EVT_param[clss_idx]

            unknownDataBase = False
            if score<args.threshold:
                unknownDataBase = True
                if cluster_model is not None:
                    preds[idx,] = clustered_classes[-1]+1 # Unknown
                else:
                    preds[idx,] = args.num_labeled_classes # Unknown            
            else:
                preds[idx,] = clss_idx # Known
            if cluster_model is not None:
                dataPointProb = clusterId_prob[idx]
                idclassEvTunknown = clusterId[idx]
                paramVecEVT = param_vec_clustered[idclassEvTunknown]
                param_1EVT = matlab.double([paramhat[0][0]])
                param_2EVT = matlab.double([paramhat[0][1]])
                rf_tree_maxEVT = np.max(dataPointProb)*args.num_trees
                cnf_treeEVT = matlab.double([rf_tree_maxEVT])          
                ConfFilterEVT = mat_eng.wblcdf(cnf_treeEVT,param_1EVT,param_2EVT,0)
            if cluster_model is not None and np.max(dataPointProb)>=0.5 and unknownDataBase:
                preds[idx,] = clustered_classes[0] # Unknown
        return preds
    
    elif args.OSR_method == 'RFThresh':
        class_id = rf_class_pred
        class_score = np.max(rf_class_prob ,axis=1)
        for idx,(clss_idx,score) in enumerate(zip(class_id,class_score)):
            paramhat = EVT_param[clss_idx]

            unknownDataBase = False
            if score<args.threshold:
                unknownDataBase = True
                if cluster_model is not None:
                    preds[idx,] = clustered_classes[-1]+1 # Unknown
                else:
                    preds[idx,] = args.num_labeled_classes # Unknown            
            else:
                preds[idx,] = clss_idx # Known
            if cluster_model is not None:
                dataPointProb = clusterId_prob[idx]
                idclassEvTunknown = clusterId[idx]
                paramVecEVT = param_vec_clustered[idclassEvTunknown]
                param_1EVT = matlab.double([paramhat[0][0]])
                param_2EVT = matlab.double([paramhat[0][1]])
                rf_tree_maxEVT = np.max(dataPointProb)*args.num_trees
                cnf_treeEVT = matlab.double([rf_tree_maxEVT])          
                ConfFilterEVT = mat_eng.wblcdf(cnf_treeEVT,param_1EVT,param_2EVT,0)
            if cluster_model is not None and ConfFilterEVT>=0.5 and unknownDataBase:
                preds[idx,] = clustered_classes[0] # Unknown
        return preds        

def extract_features(model,x):
    model.eval()
    features_model = torch.zeros((x.shape[0],512))
    #x = torch.from_numpy(data)
    with torch.no_grad():
        x = x.to(device)
        for idx in range(x.shape[0]):
            scenario_in = x[idx,:,:,:,:]
            scenario_in = rearrange(scenario_in, 'c f h w -> 1 c f h w')
            if args.model_use=='Times':
                scenario_in = rearrange(scenario_in, 'b c f h w  -> b f c h w ') 
            _,  features = model(scenario_in)
            features_model[idx,:] = features

    return features_model

def calculate_silhoutte_score(data,args):
    for n_class in range(2,10):
        KMean= KMeans(n_clusters=n_class,n_jobs=-1,n_init=20)
        KMean.fit(data)
        label=KMean.predict(data)   
        sil_score = silhouette_score(data, label) 
        print('\nSil Score for Cluster with {} is {:.4f}'.format(n_class,sil_score))  

def outlier_detection(data,args):
    if args.outlier_method == 'Isolation_Forest':
        iso = IsolationForest()
        yhat = iso.fit_predict(data)
        return  yhat != -1
    elif args.outlier_method == 'LOF':
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        # use fit_predict to compute the predicted labels of the training samples
        # (when LOF is used for outlier detection, the estimator has no predict,
        # decision_function and score_samples methods).
        y_pred = clf.fit_predict(X)
        return y_pred != -1
    elif args.outlier_method == 'OCSVM':
        clf = OneClassSVM(gamma='auto').fit(X)
        y_pred = clf.fit_predict(X)
        return y_pred != -1
    
    

def find_FNFP(predictions,unknownTrueMask,args,clustered_classes=None):
    #unknownTrueMask = np.array(unknownTrueMask,dtype=bool)
    unknownLabels = predictions[unknownTrueMask]
    knowLabels = predictions[~unknownTrueMask]
    if clustered_classes is not None:
        unknownLabelsWrongIds = unknownLabels<(clustered_classes[-1]+1)
        knownLabelsWrongIds =  knowLabels==(clustered_classes[-1]+1)
    else:
        unknownLabelsWrongIds = unknownLabels<args.num_labeled_classes
        knownLabelsWrongIds =  knowLabels==args.num_labeled_classes

    FN = np.count_nonzero(unknownLabelsWrongIds)    
    FP = np.count_nonzero(knownLabelsWrongIds)

    return FN,FP#unknownLabelsWrongIds,knownLabelsWrongIds 

def fit_cluster(data,args):
    # Elbow method
    KMean= KMeans(n_init=20)
    vis_elbow = KElbowVisualizer(KMean, k=(2,20), timings= True)
    elbow_git = vis_elbow.fit(data)        # Fit data to visualizer
    
    # Sillhoutte index
    vis_sill = KElbowVisualizer(KMean, k=(2,20),metric='silhouette', timings= True)
    sil_git = vis_sill.fit(data)     

    # Calinkski Harabasz index
    vis_CHI = KElbowVisualizer(KMean, k=(2,20),metric='calinski_harabasz', timings= True)
    chi_git = vis_CHI.fit(data)        # Fit the data to the visualizer   
    
    cluster_values =  [sil_git.elbow_value_,chi_git.elbow_value_,elbow_git.elbow_value_]
    return min(cluster_values), sil_git.elbow_score_

def clustering_data_load(args,mat_file_name):
    # Clustering load
    if args.model_use == 'AE':
        clusterig_file = 'clustering_scenarios_pure_ae' 
    else:
        clusterig_file = 'clustering_scenarios' 

    model_dir= os.path.join(args.exp_root, clusterig_file, args.model_use, str(args.model_depth))
    args.load_model_dir = model_dir
    elbow_value = loadmat(mat_file_name)['elbow']
    elbow_value = np.asarray(elbow_value[0][0])
    elbow_value = args.clustered_classes
    if args.old_clustered_classes ==0:
        modelChoose = None
    else:
        modelChoose = args.old_clustered_classes

    args.cluster_model_root = args.load_model_dir+'\\'+'{}_cluster.pth'.format(args.model_name + '_' + str(modelChoose) + '_' + str(args.clustered_classes))
    model_dir= os.path.join(args.exp_root,clusterig_file , args.model_use, str(args.model_depth))
    clusterSaveDir = args.load_model_dir+'\\cluster'+ '_' + args.SSL + '_' + str(args.num_labeled_classes) + str(modelChoose) + '_' + str(args.clustered_classes) +\
         '_'+ str(args.cluster_index)+ '_'+'.npy'

    # Choose the model
    if args.model_use == 'ResNet': 
        model = generate_model_cluster(args.model_depth, args.num_labeled_classes+ elbow_value, elbow_value) 
        model = model.to(device)
    
    elif args.model_use == 'AE':
        model = AE(num_labelled=args.num_labeled_classes,num_unlabelled=args.num_unlabeled_classes,image_size=args.img_height,model_depth=args.model_depth)
        model = model.to(device)
    else:
        model = TimeSformerCluser(
            dim = 512,
            image_height = args.img_height,        # image height
            image_width = args.img_width,        # image width  
            patch_height = 20,         # patch height
            patch_width = 20,         # patch width  
            num_frames = args.img_depth,           # number of image frames, selected out of video
            num_classes = args.num_labeled_classes + elbow_value,
            num_Unlabelledclasses = elbow_value, 
            depth = 8,
            heads = 4,
            dim_head =  64,
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )
        model = model.to(device)
    state_dict = torch.load(args.cluster_model_root)
    model.load_state_dict(state_dict,strict=False)
    for name, param in model.named_parameters(): 
       param.requires_grad = False
    
    existingClusterData = np.load(clusterSaveDir,allow_pickle=True)
    existingClusterData = existingClusterData.item()
    cluster_features = existingClusterData['features']
    cluster_labels = existingClusterData['labels']
    cluster_true_labels = existingClusterData['true_labels']
    
    #X_train, X_test, y_train, y_test = train_test_split(cluster_features, cluster_labels, test_size=0.25)
    
    
    return {'cluster_deep_model':model,'cluster_features':cluster_features,'cluster_labels':cluster_labels,'cluster_true_labels':cluster_true_labels}

# To Do:
# 1. Add multiple mat files for unkown data
# 2. Add the correct cluster model
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='OSR',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=2, type=int)
    parser.add_argument('--clustered_classes', default=2, type=int)
    parser.add_argument('--num_labeled_classes', default=6, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/Scenarios')
    parser.add_argument('--ssl_dir', type=str, default='./data/experiments/selfsupervised_learning_scenario_barlow_twins/')
    parser.add_argument('--model_name', type=str, default='ssl_labelled')
    parser.add_argument('--dataset_name', type=str, default='scenarios', help='options: highd, roundabout')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_use', type=str, default='ResNet')
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--UMAP', type=bool, default=False)
    parser.add_argument('--img_width', type=int, default=120)
    parser.add_argument('--img_height', type=int, default=120)
    parser.add_argument('--img_depth', type=int, default=4)
    parser.add_argument('--num_trees', type=int, default=250)
    parser.add_argument('--outlier_method', type=str, default='Isolation_Forest')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_root', type=str, default='./data/experiments/supervised_learning_scenarios/') 
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--osr_filter', type=float, default=0.5)
    parser.add_argument('--cluster_trigger', type=int, default=np.inf,help='np.inf or any number as input') # help this can be buffer 
    parser.add_argument('--SSL', type=str, default='Barlow')
    parser.add_argument('--OSR_method', type=str, default='EVT')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--split_strategy', type=str, default='equal') # 'equal' or 'unequal'
    parser.add_argument('--cluster_index', type=int, default=1)
    parser.add_argument('--cluster_cycle', type=int, default=1)
    parser.add_argument('--old_clustered_classes', type=int, default=0)
    parser.add_argument('--buffer_select', type=int, default=32,help='used to select the buffer stored can be fixed sized buffer or \
                        complete buffer with infinite storage')
    parser.add_argument('--osr_root', type=str, default='./data/experiments/open_set_recognition/ResNet/18/') 
    parser.add_argument('--outier_required', type=bool, default=True)
    mat_eng = matlab.engine.start_matlab()
  
    args = parser.parse_args()
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name, args.model_use, str(args.model_depth))
    rf_evt_for = os.path.join(args.model_root, args.model_use, str(args.model_depth))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.save_dir = model_dir
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    
    if args.model_use == 'ResNet' or  args.model_use == 'AE':
        args.model_name = args.model_name + '_resnet_' + str(args.model_depth) + '_' +str(args.SSL)+ '_' + str(args.num_labeled_classes) 
    else:
        args.model_name = args.model_name + '_times_'  + str(args.SSL)+ '_' + str(args.num_labeled_classes)
    if args.clustered_classes is not None:
        args.mat_save_name = args.model_name + '_' + str(args.clustered_classes+args.old_clustered_classes) + '_' + str(args.num_unlabeled_classes)
    else:
        args.mat_save_name = args.model_name + '_' + str(None) + '_' + str(args.num_unlabeled_classes)
    args.model_dir = rf_evt_for+'/'+'{}.pth'.format(args.model_name)
    args.rf_save_dir = rf_evt_for+'/'+'{}'.format(args.model_name) +'_RF_{}'.format(args.num_trees)  + '_' + str(args.num_labeled_classes)+ '/' + 'RF.pkl' 
    args.evt_save_dir = rf_evt_for+'/'+'{}'.format(args.model_name) +'_EVT_{}'.format(args.num_trees) + '_' + str(args.num_labeled_classes)+ '/' + 'EVT_model_'+ args.model_name + '.json'
   
    
    if args.clustered_classes is not None:
        labelled_plus_clustered_classes = args.num_labeled_classes + args.old_clustered_classes +  args.clustered_classes
        num_classes = labelled_plus_clustered_classes + args.num_unlabeled_classes
        clustered_classes = list(range(args.num_labeled_classes, labelled_plus_clustered_classes))
        known_classes = list(range(labelled_plus_clustered_classes))
        unknown_classses = list(range(labelled_plus_clustered_classes, num_classes))
        unknown_classses = unknown_classses[:args.num_unlabeled_classes]
    else:
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        unknown_classses = list(range(args.num_labeled_classes, num_classes))
        unknown_classses = unknown_classses[:args.num_unlabeled_classes]
        known_classes = list(range(args.num_labeled_classes))  
    
    
    if args.clustered_classes is not None:
        if args.old_clustered_classes == 0:
            args.mat_load_dir = args.model_name + '_' + str(None) + '_' + str(args.num_unlabeled_classes)  # str(args.clusterd_classes)          
            matfileToLoad = args.mat_load_dir + '_' + str(args.buffer_select) + '_' 
            mat_file_name = args.osr_root+'/'+'{}.mat'.format(matfileToLoad) 
        else:
            # get all the mat files in the folder
            mat_file_name = [f for f in os.listdir(model_dir) if f.endswith('.mat')]
            mat_file_name = [os.path.join(model_dir, f) for f in mat_file_name]
        if args.old_clustered_classes==0:
            args.mat_load_dir_elbow = args.model_name + '_' + str(None) + '_' + str(args.num_unlabeled_classes)  # str(args.clusterd_classes)          
        else:
            args.mat_load_dir_elbow = args.model_name + '_' + str(args.old_clustered_classes) + '_' + str(args.clustered_classes)  # str(args.clusterd_classes)          
        matfileToLoad_elbow = args.mat_load_dir_elbow + '_' + str(args.buffer_select) + '_' 
        mat_file_name_elbow = args.osr_root+'/'+'{}.mat'.format(matfileToLoad_elbow)         
        # Train an outlier model
        unlabeled_eval_loader = ScenarioLoader(root=mat_file_name, batch_size=args.batch_size, split='unknownTrain', aug='once', shuffle=False)
        cluster_data = clustering_data_load(args,mat_file_name_elbow)    
        cluster_deep_model = cluster_data['cluster_deep_model']
        cluster_deep_model = cluster_deep_model.to(device)
        inlierDataTraining = []
        cluster_true_labels = []
        for batch_idx, (Cx,  Clabel, idx) in tqdm(enumerate((unlabeled_eval_loader))): 
            Cx = Cx.to(device)
            if not args.model_use == 'AE':
                _,_,features = cluster_deep_model(Cx)
            else:
                _,features,_ = cluster_deep_model(Cx)
            inlierDataTraining.append(features.cpu().detach().numpy())
            cluster_true_labels.append(Clabel.cpu().detach().numpy())
        
        inlierDataTraining = np.concatenate(inlierDataTraining, axis=0)
        cluster_true_labels = np.concatenate(cluster_true_labels, axis=0)
        inlierDatFilterdIdx = [True if ctl in clustered_classes else False for ctl in cluster_true_labels]
        inlierDatFilterdIdx = np.array(inlierDatFilterdIdx)
        cluster_true_labelsCopyDelete = cluster_true_labels.copy()
        cluster_true_labelsCopyDelete = cluster_true_labelsCopyDelete[inlierDatFilterdIdx]
        uniqueClusterLabels = np.unique(cluster_true_labelsCopyDelete)
        cluster_true_labelsCopyDelete = cluster_true_labelsCopyDelete - uniqueClusterLabels[0]
        inlierDataTraining = inlierDataTraining[inlierDatFilterdIdx,:]
        cluster_outlier_model = train_RF(inlierDataTraining,cluster_true_labelsCopyDelete,args)
        
  
        # Train the evt model
        unlabeledEVT_eval_loader = ScenarioLoader(root=mat_file_name, batch_size=args.batch_size, split='unknownTest', aug=None, shuffle=False)
        inlierDataTrainingEVT = []
        cluster_true_labelsEVT = []
        for batch_idx, (Cx,  Clabel, idx) in tqdm(enumerate((unlabeledEVT_eval_loader))): 
            Cx = Cx.to(device)
            if not args.model_use == 'AE':
                _,_,features = cluster_deep_model(Cx)
            else:
                _,features,_ = cluster_deep_model(Cx)
            inlierDataTrainingEVT.append(features.cpu().detach().numpy())
            cluster_true_labelsEVT.append(Clabel.cpu().detach().numpy())
        inlierDataTrainingEVT = np.concatenate(inlierDataTrainingEVT, axis=0)
        cluster_true_labelsEVT = np.concatenate(cluster_true_labelsEVT, axis=0)  
        uniqueClusterLabelsEVT, uniClusterCount = np.unique(cluster_true_labelsEVT,return_counts=True)
        countGreaterIndex = np.where(uniClusterCount > 5)    # keep 50
        uniqueClusterLabelsEVT = uniqueClusterLabelsEVT[countGreaterIndex] 
        #uniqueClusterLabelsEVT = [6,7]
        param_vec_clustered = train_evt(inlierDataTrainingEVT,cluster_true_labelsEVT,cluster_outlier_model,uniqueClusterLabelsEVT)  
    else:
        param_vec_clustered = None
        cluster_deep_model = None
        cluster_outlier_model = None
        clustered_classes = None
        
    # load RF
    with open(args.rf_save_dir,'rb') as fp:
        RF_clf = cPickle.load(fp)
        
    # load EVT
    with open(args.evt_save_dir) as json_file:
        EVT = json.load(json_file)
    EVT_param = EVT['EVT_Param']
    # Choose the model
    if args.model_use == 'ResNet' or  args.model_use == 'AE':    
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

    
    classes_chosen = known_classes + unknown_classses
    #test_loader = ScenarioLoader(root=args.dataset_root, batch_size=args.batch_size, split='testplusunknown', shuffle=True, aug=None, target_list = classes_chosen)
    if args.split_strategy == 'equal':
        test_loader = ScenarioLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='testplusunknown', aug=None,
                        shuffle=True, labeled_list=known_classes, unlabeled_list=unknown_classses, new_labels=None)
    elif args.split_strategy == 'unequal':
        test_loader = ScenarioLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='testplusunknownUE', aug=None,
                        shuffle=True, labeled_list=known_classes, unlabeled_list=unknown_classses, new_labels=None)        
    # Testing the OSR solution
    preds = np.array([])
    gTruth = np.array([])
    targets=np.array([])
    maskUnknownFull = np.array([],dtype=bool)
    unknown_input_collect = torch.empty([])
    unknown_labels_collect = np.array([])
    unknown_Tlabels_collect = np.array([])
    unknown_psuedolabels_collect = np.array([])

    unknown_features_collect = torch.empty([])
    unknownClassesTrueNumber = 0
    knownClassesTrueNumber = 0
    totalInstances = 0
    totalEmergClassesInstance = 0
    
    accuracyValues = {'mNew':[],'fNew':[]}
    
    
    # # Debug for existing class check
    #!!! DELETE !!!
    # if args.clustered_classes is not None:
    #     gTruthPlot = []
    #     clusterTempFeaturesCollect = []
    #     cluster_deep_model.train()

    #     for batch_idx, (x,  label, idx) in tqdm(enumerate((test_loader))):
            
    #         x = x.to(device)
    #         _,_,clusterTempFeatures = cluster_deep_model(x)
    #         clusterTempFeatures = clusterTempFeatures.detach()
    #         label = label.cpu().numpy()
    #         clusterTempFeatures = clusterTempFeatures.cpu().numpy()
    #         for i in known_classes[args.num_labeled_classes:]:
    #             #label[label == i] = 7
    #             #visualize_scenarios(x[label==4].cpu(),3,args)

    #             gTruthPlot = np.append(gTruthPlot, label[label==i])
    #             clusterTempFeaturesCollect.append(clusterTempFeatures[label==i])
    #     #gTruthPlot[:]  = 7
    #     clusterTempFeaturesCollect = np.concatenate(clusterTempFeaturesCollect, axis=0)
    #     clusterTempFeaturesCollect = np.concatenate((inlierDataTraining,clusterTempFeaturesCollect), axis=0)
    #     gTruthPlot = np.concatenate((cluster_true_labelsCopyDelete,gTruthPlot), axis=0)
        
    #     plot_umap(clusterTempFeaturesCollect, gTruthPlot, 2000000, args) 
        
    # # !!! DELETE END !!!
    for batch_idx, (x,  label, idx) in tqdm(enumerate((test_loader))):
        json_osr_file_name = args.mat_save_name  + '_' + str(batch_idx) + '_.json'
        json_acc_file_name = args.mat_save_name + '_' + str(batch_idx) + '_Acc_.json'
        mat_file_name = args.mat_save_name  + '_' + str(batch_idx) + '_.mat'
        result_save_dir = model_dir  + '\\' + json_osr_file_name
        acc_save_dir = model_dir + '\\' + json_acc_file_name

        data_save_dir = model_dir + '\\' +  mat_file_name
        pred_label = osr_single_scenario(model, RF_clf, EVT_param, x, mat_eng, args, cluster_deep_model,cluster_outlier_model,clustered_classes,label,param_vec_clustered)
        
        # OSR predictions save
        preds = np.append(preds, pred_label)
        # True ground truth save
        gTruth = np.append(gTruth, label)
        # True labels saved to calculate total Emering classes
        true_labels =  np.copy(label.cpu().numpy())
        if args.clustered_classes is not None:
            for i in clustered_classes:
                label[label == i] = clustered_classes[0]

        totalNumberofInstances = x.shape[0]
        totalInstances += totalNumberofInstances
        if args.clustered_classes is not None:
            label[label>=labelled_plus_clustered_classes] = labelled_plus_clustered_classes
            knownClassesTrueNumber+=np.count_nonzero(label<labelled_plus_clustered_classes)
            totalEmergClassesTrueNumber = np.count_nonzero(true_labels>=labelled_plus_clustered_classes)
            unknownClassesTrueNumber+=np.count_nonzero(label>=labelled_plus_clustered_classes)
            maskUnknown = label>=labelled_plus_clustered_classes
        else:
            label[label>=args.num_labeled_classes] = args.num_labeled_classes
            knownClassesTrueNumber+=np.count_nonzero(label<args.num_labeled_classes)
            totalEmergClassesTrueNumber = np.count_nonzero(true_labels>=args.num_labeled_classes)
            unknownClassesTrueNumber+=np.count_nonzero(label>=args.num_labeled_classes)
            maskUnknown = label>=args.num_labeled_classes
        a,b = torch.unique(label, return_counts=True)    
        c,d = np.unique(pred_label, return_counts=True)
        print('True classes: {} known class true counts {}'.format(a, b))
        print('Pred classes: {} known class pred counts {}'.format(c, d))
        
        
        totalEmergClassesInstance += totalEmergClassesTrueNumber
        maskUnknownFull = np.append(maskUnknownFull,maskUnknown)
        FN, FP = find_FNFP(pred_label,maskUnknown,args,clustered_classes) 
        if totalEmergClassesTrueNumber is not 0:
            accuracyValues['mNew'].append(FN*100/totalEmergClassesTrueNumber)
            accuracyValues['fNew'].append(FP*100/(totalNumberofInstances-totalEmergClassesTrueNumber))
         
            print('\nM_new {}'.format(FN*100/totalEmergClassesTrueNumber))
            print('\nF_new {}'.format(FP*100/(totalNumberofInstances-totalEmergClassesTrueNumber)))
              
        print('\nTotal no of known classes {}'.format(knownClassesTrueNumber))
        print('\nTotal no of unknown classes {}'.format(unknownClassesTrueNumber))
        print('\nF1 score',f1_score(label,pred_label,average='weighted'))
        print('\nF1 score',f1_score(label,pred_label,average=None))
        
        # Ground truth save
        targets = np.append(targets, label)
        precision,recall,fscore,_=precision_recall_fscore_support(targets,preds,average='weighted')
        if args.clustered_classes is not None:
            print('\nMacro F score for the split {}:{} in method {} is {}'.format(labelled_plus_clustered_classes,
                                                                                        args.num_unlabeled_classes,args.OSR_method,fscore))
        else:
            print('\nMacro F score for the split {}:{} in method {} is {}'.format(args.num_labeled_classes,
                                                                                        args.num_unlabeled_classes,args.OSR_method,fscore))
        osr_data = {'OSR_Data':fscore,'F_score':list(f1_score(label,pred_label,average=None))}
        # Make sure this is right
        if args.clustered_classes is not None:
            raw_unknown_samples = x[pred_label==labelled_plus_clustered_classes,:,:,:,:]
        else:
            raw_unknown_samples = x[pred_label==args.num_labeled_classes,:,:,:,:]
        with open(result_save_dir, 'w') as outfile:
            json.dump(osr_data, outfile)
        with open(acc_save_dir, 'w') as outfile:
            json.dump(accuracyValues, outfile)  
        try:                   
            if unknown_input_collect.shape==torch.Size([]):
                unknown_input_collect = raw_unknown_samples
            else:
                unknown_input_collect = torch.cat((unknown_input_collect, raw_unknown_samples),axis=0)
        except:
            aa = 0
        if args.clustered_classes is not None:
            unknown_labels_collect = np.append(unknown_labels_collect, label[pred_label== labelled_plus_clustered_classes,])
            unknown_Tlabels_collect = np.append(unknown_Tlabels_collect, true_labels[pred_label== labelled_plus_clustered_classes,])
            unknown_psuedolabels_collect = np.append(unknown_psuedolabels_collect, pred_label[pred_label== labelled_plus_clustered_classes,])

        else:
            unknown_labels_collect = np.append(unknown_labels_collect, label[pred_label== args.num_labeled_classes,])
            unknown_Tlabels_collect = np.append(unknown_Tlabels_collect, true_labels[pred_label== args.num_labeled_classes,])
            unknown_psuedolabels_collect = np.append(unknown_psuedolabels_collect, pred_label[pred_label== args.num_labeled_classes,])

        print('\nUnknown samples size {}'.format(unknown_input_collect.shape[0]))

        if raw_unknown_samples.shape[0]>0:
            
            if unknown_features_collect.shape==torch.Size([]):
                unknown_features_collect = extract_features(model,raw_unknown_samples)
            else:
                unknown_features_collect = torch.cat((unknown_features_collect, extract_features(model,raw_unknown_samples)),axis=0)
        
        
        
        # Unknown data base based clustering
        if unknown_input_collect.shape[0]>args.cluster_trigger:
            
            if args.outier_required:
                mask = outlier_detection(unknown_features_collect,args)
            else:
                mask = np.ones(unknown_input_collect.shape[0],dtype=bool)
            U = umap.UMAP(n_components = 2)
            embedding2 = U.fit_transform(unknown_features_collect[mask,:],)    
            fig, ax = plt.subplots(1, figsize=(14, 10))
            plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=unknown_Tlabels_collect[mask], cmap='Spectral')
            savename = 'Test.png'
            plt.savefig(savename)
            #calculate_silhoutte_score(unknown_features_collect[mask, :],args)
            elbow_value, elbow_score = fit_cluster(embedding2,args)
            if elbow_value is not None:
                KMean= KMeans(n_clusters=elbow_value,n_jobs=-1,n_init=20)
                KMean.fit(unknown_features_collect[mask, :])
                label=KMean.predict(unknown_features_collect[mask, :])   
                plt.clf()
                plt.cla()   
                plt.close()
                fig, ax = plt.subplots(1, figsize=(14, 10))
                print('\n Clustering Accuracy {:.4f}'.format(cluster_acc(unknown_Tlabels_collect[mask].astype(int), label.astype(int))))
                print('\n Elbow value {:.4f}'.format(elbow_value))
                print('\n Actual Clusters {}'.format(len(np.unique(unknown_Tlabels_collect[mask]))))
                plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=label, cmap='Spectral')
                savename = 'Test1.png'
                plt.savefig(savename)
                preds=np.array([])
                save_mat={}
                save_mat['unknown_input_collect'] = unknown_input_collect[mask,:,:,:,:].cpu().numpy()
                save_mat['unknown_labels_collect'] = unknown_labels_collect[mask]
                save_mat['unknown_Tlabels_collect'] = unknown_Tlabels_collect[mask]
                save_mat['elbow'] = elbow_value
                save_mat['actual_clusters'] =  len(np.unique(unknown_Tlabels_collect[mask]))
                
                io.savemat(data_save_dir,save_mat)
            preds = np.array([])
            gTruth = np.array([])
            targets=np.array([])
            unknown_input_collect = torch.empty([])
            unknown_labels_collect = np.array([])
            unknown_Tlabels_collect = np.array([])
            unknown_features_collect = torch.empty([])
            break
    del x
    del test_loader     
    # Unlimited memory for clustering
    if args.cluster_trigger == np.inf:
            if args.outier_required:
                mask = outlier_detection(unknown_features_collect,args)
            else:
                mask = np.ones(unknown_input_collect.shape[0],dtype=bool)
            if mask.shape[0]>0:
                U = umap.UMAP(n_components = 2)
                embedding2 = U.fit_transform(unknown_features_collect[mask,:],)    
                elbow_value, elbow_score = fit_cluster(embedding2,args)
                if elbow_value is not None:
                    KMean= KMeans(n_clusters=elbow_value,n_jobs=-1,n_init=20)
                    KMean.fit(unknown_features_collect[mask, :].cpu().numpy())
                    label=KMean.predict(unknown_features_collect[mask, :].cpu().numpy())   
                    plt.clf()
                    plt.cla()   
                    plt.close()
                    fig, ax = plt.subplots(1, figsize=(14, 10))
                    print('\n Clustering Accuracy {:.4f}'.format(cluster_acc(unknown_Tlabels_collect[mask].astype(int), label.astype(int))))
                    print('\n Elbow value {:.4f}'.format(elbow_value))
                    print('\n Actual Clusters {}'.format(args.num_unlabeled_classes))
                    plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=label, cmap='Spectral')
                    savename = 'Test1.png'
                    plt.savefig(savename)
            
            
            
                save_mat={}
                save_mat['unknown_input_collect'] = unknown_input_collect[mask,:,:,:,:].cpu().numpy()
                save_mat['unknown_labels_collect'] = unknown_labels_collect[mask]
                save_mat['unknown_Tlabels_collect'] = unknown_Tlabels_collect[mask]
                save_mat['elbow'] = elbow_value
                save_mat['actual_clusters'] =  len(np.unique(unknown_Tlabels_collect[mask]))
                io.savemat(data_save_dir,save_mat)
                U = umap.UMAP(n_components = 2)
                embedding2 = U.fit_transform(unknown_features_collect[mask,:].cpu().numpy(),)    
                fig, ax = plt.subplots(1, figsize=(14, 10))
                plt.scatter(embedding2[:, 0], embedding2[:, 1], s= 5, c=unknown_Tlabels_collect[mask], cmap='Spectral')
                savename = 'Test.png'
                plt.savefig(savename)
                
                FN, FP = find_FNFP(preds,maskUnknownFull,args) 
                accuracyValues['mNew'].append(FN*100/totalEmergClassesInstance)
                accuracyValues['fNew'].append(FP*100/(totalInstances-totalEmergClassesInstance))
                
                print('\nM_new {}'.format(FN*100/totalEmergClassesInstance))
                print('\nF_new {}'.format(FP*100/(totalInstances-totalEmergClassesInstance)))
                gTruthConfusionMatrixCalc = gTruth.copy()
                if args.clustered_classes is not None:
                    gTruthConfusionMatrixCalc[gTruthConfusionMatrixCalc>=args.num_labeled_classes] = args.num_labeled_classes+1
                else:
                    gTruthConfusionMatrixCalc[gTruthConfusionMatrixCalc>=args.num_labeled_classes] = args.num_labeled_classes
                cm = confusion_matrix(gTruthConfusionMatrixCalc, preds,normalize='true')
                if args.clustered_classes is not None:
                    class_names = list(range(args.num_labeled_classes+1))
                    class_names = [str(i) for i in class_names]
                else:
                    class_names = list(range(args.num_labeled_classes+1))
                    class_names = [str(i) for i in class_names]
                #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                #disp.plot() 
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.imsave('confusion_matrix.png', cm)
                plt.savefig('ConfusionMatrix.png')
                #calculate_silhoutte_score(unknown_features_collect[mask, :],args)

 
       





      

