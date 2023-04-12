import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection
import random
import train_utils
import torchxrayvision as xrv

import nih_dataset

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('-name', type=str, default="test") #pretrain_densenet
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--dataset', type=str, default="nih")
    parser.add_argument('--dataset_dir', type=str, default="imgdata")
    parser.add_argument('--model', type=str, default="densenet")#pretrain_densenet -" pretrain"
    parser.add_argument('--seed', type=int, default=6759, help='')
    parser.add_argument('--cuda', type=bool, default=True, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--threads', type=int, default=4, help='') #torch.utils.data.DataLoader(num_workers=cfg.threads,)
    parser.add_argument('--taskweights', type=bool, default=False, help='')# taskweights for BCE loss
    parser.add_argument('--featurereg', type=bool, default=False, help='')
    parser.add_argument('--weightreg', type=bool, default=False, help='')
    parser.add_argument('--data_aug', type=bool, default=True, help='')
    parser.add_argument('--data_aug_rot', type=int, default=45, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--label_concat', type=bool, default=False, help='')
    parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
    parser.add_argument('--labelunion', type=bool, default=False, help='')


    #choice loss function and optimizer
    parser.add_argument('--loss_func', type=str, default='AUCM_MultiLabel', help='')        #BCEWithLogitsLoss or AUCM_MultiLabel or label_smoothing
    parser.add_argument('--optimizer', type=str, default='PESG', help='')                   #adam or PESG
    
    #only for AUCM_MultiLabel and PESG
    parser.add_argument('--update_lr', type=bool, default=False, help='')                   #AUCM_MultiLabel update lr
    parser.add_argument('--update_regularizer', type=bool, default=False, help='')          #AUCM_MultiLabel update lr and update update_regularizer #DO NOT USE !!!
    parser.add_argument('--decay_factor', type=float, default=2, help='')                   #new = old/decay_factor
    parser.add_argument('--decay_epoch', type=int, default=10, help='')                     #epoch%decay_epoch == 0 do update

    parser.add_argument('--margin_AUCloss', type=float, default=1, help='')                 #can also be tuned in [0.5, 1.0] for better performance   
    parser.add_argument('--PESG_momentum', type=float, default=0, help='')                  #is similar to SGD-momentum and you can choose value in [0, 0.9]


    print(os.getcwd())
    cfg = parser.parse_args()
    print(cfg)


    data_aug = None
    if cfg.data_aug:
        data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                                translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                                scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
            torchvision.transforms.ToTensor()
        ])
    print(data_aug)

    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])#Resizer(512)#


    datas = []
    datas_names = []
    #in our case will only use 2 dataset [nih,cheXpert]
    if "nih" in cfg.dataset:
        dataset = nih_dataset.NIH_Dataset(
            imgpath=cfg.dataset_dir + "/images-NIH-224", #we use smaller data set
            transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
        datas.append(dataset)
        datas_names.append("nih")
    if "google" in cfg.dataset:
        print("Yizao's show time")
    if "chex" in cfg.dataset:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
            csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
            transform=transforms, data_aug=data_aug, unique_patients=False)
        datas.append(dataset)
        datas_names.append("chex")
    
    print(datas)

    if cfg.labelunion:
        newlabels = set()
        for d in datas:
            newlabels = newlabels.union(d.pathologies)
        newlabels.remove("Support Devices")
        print(list(newlabels))
        for d in datas:
            xrv.datasets.relabel_dataset(list(newlabels), d)
    else:
        for d in datas:
            xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)


    #cut out training sets
    train_datas = []
    test_datas = []
    for i, dataset in enumerate(datas):
        
        # give patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
            
        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
        print("data_distribution", dataset)      
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
        test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
        
        train_datas.append(train_dataset)
        test_datas.append(test_dataset)
        
    if len(datas) == 0:
        raise Exception("no dataset")
    elif len(datas) == 1:
        #print('yep')
        train_dataset = train_datas[0]
        test_dataset = test_datas[0]
    else:
        print("merge datasets")
        train_dataset = xrv.datasets.Merge_Dataset(train_datas)
        test_dataset = xrv.datasets.Merge_Dataset(test_datas)

    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("train_dataset.labels.shape", train_dataset.labels.shape)
    print("test_dataset.labels.shape", test_dataset.labels.shape)
    print("train_dataset",train_dataset)
    print("test_dataset",test_dataset)

    # create models
    if "densenet" in cfg.model:
        model = xrv.models.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
                                    **xrv.models.get_densenet_params(cfg.model)) 
    elif "resnet101" in cfg.model:
        model = torchvision.models.resnet101(num_classes=train_dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif "resnet50" in cfg.model:
        model = torchvision.models.resnet50(num_classes=train_dataset.labels.shape[1], pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif "EfficientNet_V2" in cfg.model:
        model = torchvision.models.efficientnet_v2_s(num_classes=train_dataset.labels.shape[1])
        #patch for single channel
        print(model)
        model.features[0][0] = torch.nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif "pretrain_densenet" in cfg.model:
        model_path = "train_output/nih-densenet-test-best.pt"
        model = torch.load(model_path)

    else:
        raise Exception("no model")
    
    train_utils.train(model, train_dataset, cfg)


    

    print("Done")