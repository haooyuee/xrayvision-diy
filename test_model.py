import os,sys
import os,sys,inspect
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection
import train_utils
import random
import torchxrayvision as xrv




def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        results = []
        #result["Pathology"] = model.pathologies[i]
        task_aucs = []
        for task in range(len(task_targets)):
            result = {}
            result["Pathology"] = test_dataset.pathologies[task]
            #print(test_dataset.pathologies[task])
            if len(np.unique(task_targets[task]))> 1:
                #print(task_targets[task])
                #print(task_outputs[task])
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #task_acc = sklearn.metrics.accuracy_score(task_targets[task], task_outputs[task] > 0.5)#add
                #task_f1 = sklearn.metrics.f1_score(task_targets[task], task_outputs[task] > 0.5)#add
                result["AUC"] = task_auc
                result["Acc"] = sklearn.metrics.accuracy_score(task_targets[task], task_outputs[task] > 0.5)
                result["F1"] = sklearn.metrics.f1_score(task_targets[task], task_outputs[task] > 0.5)

                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)
            results.append(result)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets, results

def plot(auc_scores, save_path=None, name = None):
    mean_auc = np.mean(auc_scores)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(auc_scores)), auc_scores)
    ax.set_xticks(range(len(auc_scores)))
    ax.set_xticklabels(['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
                        'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
                        'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'], rotation=90)
    ax.set_ylabel('AUC score')
    ax.axhline(y=mean_auc, color='black', linestyle='--')
    plt.bar_label(ax.containers[0], label_type='center')
    ax.set_title('AUC scores for ' + name)
    plt.show()
    print('save plot')
    fig.savefig(save_path+'/'+ name, bbox_inches = 'tight')





if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="aucm_1_test")
    parser.add_argument('--save_path', type=str, default="resultat")
    parser.add_argument('--model_path', type=str, default="train_output/nih-densenet-densenet_aucm_0412_lr0005_mo07_40epoch-best.pt")
    parser.add_argument('--seed', type=int, default=6759, help='')

    cfg = parser.parse_args()
    print(cfg)

    datas = []
    datas_names = []

    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
    data_aug = None
    dataset = xrv.datasets.NIH_Dataset(
        imgpath="imgdata/images-NIH-224",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("nih")
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)
    print("datas_names", datas_names)
    print("--------------------------")
    print("data_distribution", datas)

    #cut out training sets
    train_datas = []
    test_datas = []
    for i, dataset in enumerate(datas):
            
        # give patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
                
        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
            
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
        test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
            
        train_datas.append(train_dataset)
        test_datas.append(test_dataset)

    model = torch.load(cfg.model_path)

    # Dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=8,
                                                shuffle=None,
                                                num_workers=0, pin_memory=False)
    auc, task_aucs, task_outputs, task_targets, result_all_task = valid_test_epoch("test", 0, model, "cuda", test_loader, torch.nn.BCEWithLogitsLoss(), limit=10000000)
    result_all_task_df = pd.DataFrame(result_all_task)
    print(result_all_task_df)

    result_all_task_df.to_csv(cfg.save_path +'/'+ cfg.name + '.csv')

    auc_scores = [x for x in task_aucs if str(x) != 'nan']
    auc_scores = [float('{:.2f}'.format(i)) for i in auc_scores]

    print(auc_scores)
    plot(auc_scores,cfg.save_path,cfg.name)
    print('Done')

