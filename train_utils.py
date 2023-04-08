import json
import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join

import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import torchxrayvision as xrv

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
#from tqdm.auto import tqdm

import losses
from libauc.optimizers import PESG, Adam




def train(model, dataset, cfg):
    print("Our config:")
    pprint.pprint(cfg)
        
    dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
    
    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    print(cfg.output_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset    
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.01,test_size=0.03, random_state=cfg.seed) #0.03 for test
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    valid_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    #print(model)
    #DIY loss function
    if cfg.loss_func == None:
        raise ValueError('loss function is not defined')
    else:
        #Binary
        #default loss function
        if cfg.loss_func == 'BCEWithLogitsLoss':#Binary Cross Entropy with LogitsLoss
            criterion = torch.nn.BCEWithLogitsLoss()#https://blog.csdn.net/qq_22210253/article/details/85222093
        #elif cfg.loss_func == 'BCELoss': # Binary Cross Entropy
        #    criterion = torch.nn.BCELoss()

        #Not Binary
        elif cfg.loss_func == 'CrossEntropyLoss':
            criterion = torch.nn.CrossEntropyLoss()
            raise ValueError('CrossEntropyLoss not function')
        elif cfg.loss_func == 'AUCM_MultiLabel':
            n_positive = np.nansum(train_loader.dataset.labels, axis=0)
            n_positive = n_positive[n_positive != 0]#remove 0 value
            n_data = train_loader.dataset.labels.shape[0]
            imratio = n_positive/n_data
            imratio = torch.from_numpy(imratio).to(device).float()
            print("imratio")
            print(imratio)
            #raise ValueError('test')
            criterion = losses.AUCM_MultiLabel_V1(num_classes = 14, imratio=imratio, device=device)
        else:
            raise ValueError('invalid loss function')
    print('criterion : ')
    print(criterion)

    #DIY optimizer
    if cfg.optimizer == None:
        raise ValueError('optimizer is not defined')
    else:
        #default optimizer
        if cfg.optimizer == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
        elif cfg.optimizer == 'PESG':
            optim = PESG(model,
                 loss_fn=criterion, 
                 lr=cfg.lr, 
                 margin=1, 
                 epoch_decay=2e-3, 
                 weight_decay=1e-5)
        ##########################
        # can add more optimizer #
        ##########################
        else:
            raise ValueError('invalid optimizer')
    print('optim : ')
    print(cfg.optimizer)

    

    
    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    
    model.to(device)

    #save train process
    logger = dict()
    logger['train_losses'] = []
    logger['eval_losses'] = []
    logger['eval_auc'] = []

    for epoch in range(start_epoch, cfg.num_epochs):
        #add hyper update
        if cfg.loss_func == 'AUCM_MultiLabel':
            if epoch%cfg.decay_epoch == 0:
                if cfg.update_lr :
                    optim.update_lr(decay_factor=cfg.decay_factor)
                elif cfg.update_regularizer:
                    optim.update_regularizer(decay_factor=cfg.decay_factor)

        avg_loss  = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               valid_loader=valid_loader,
                               criterion=criterion)
        
        auc_valid, valid_loss = valid_test_epoch(cfg=cfg,
                                name='Valid',
                                epoch=epoch,
                                model=model,
                                device=device,
                                data_loader=valid_loader,
                                criterion = criterion
                                )
        
        #logger
        logger['train_losses'].append(avg_loss.item())
        logger['eval_losses'].append(valid_loss.item())
        logger['eval_auc'].append(auc_valid.item())

        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        torch.save(model, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

    logger['final_train_losses'] = avg_loss.item()
    logger['final_eval_losses'] = valid_loss.item()
    logger['final_eval_auc'] = auc_valid.item()
 
    save_logs(logger, cfg.output_dir, str(dataset_name))
    return metrics, best_metric, weights_for_best_validauc


def save_logs(dictionary, log_dir, exp_id):
    log_dir = os.path.join(log_dir, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    # Log arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(dictionary, f, indent=2)
    

def BCELogits_loss(cfg, device, targets, outputs, criterion, model, weights = None):
    loss = torch.zeros(1).to(device).float()  
    for task in range(targets.shape[1]):
        task_output = outputs[:,task]
        task_target = targets[:,task]
        mask = ~torch.isnan(task_target)#18 task in model but 14 task in data -> masked 4 nan data
        task_output = task_output[mask]
        #print('task_output')
        #print(task_output)
        task_target = task_target[mask]
        #print('task_target')
        #print(task_target)
        if len(task_target) > 0:
            task_loss = criterion(task_output.float(), task_target.float())
            #print('task_loss' + str(task_loss))
            if cfg.taskweights:
                    loss += weights[task]*task_loss
            else:
                loss += task_loss
        
    # <- is not use in our task ->
    # here regularize the weight matrix when label_concat is used
    if cfg.label_concat_reg:
        if not cfg.label_concat:
            raise Exception("cfg.label_concat must be true")
        weight = model.classifier.weight
        num_labels = len(xrv.datasets.default_pathologies)
        num_datasets = weight.shape[0]//num_labels
        weight_stacked = weight.reshape(num_datasets,num_labels,-1)
        label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
        for task in range(num_labels):
            dists = torch.pdist(weight_stacked[:,task], p=2).mean()
            loss += label_concat_reg_lambda*dists
    # <- is not use in our task ->

    loss = loss.sum()
    return loss

def train_epoch(cfg, epoch, model, device, train_loader, valid_loader, optimizer, criterion, limit=20000): #change limit
    model.train()
    weights = None
    best_val_auc = 0 

    if cfg.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)

    
    avg_loss = []
    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)
        outputs = model(images)
        
        if cfg.loss_func == 'BCEWithLogitsLoss':
            loss = BCELogits_loss(cfg, device, targets, outputs, criterion, model, weights)
        elif cfg.loss_func == 'AUCM_MultiLabel':
            mask = ~torch.isnan(targets)
            mask_output = outputs[mask]
            mask_target = targets[mask]
            mask_output = mask_output.view(outputs.size(0), -1)
            mask_target = mask_target.view(targets.size(0), -1)
            mask_output = torch.sigmoid(mask_output)
            loss = criterion(mask_output, mask_target, auto=False)
        if cfg.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()    
        if cfg.weightreg:
            loss += model.classifier.weight.abs().sum()
        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

   
    return np.mean(avg_loss)

def valid_test_epoch(cfg, name, epoch, model, device, data_loader, criterion, limit=20000): #change limit
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

            #LOSS
            if cfg.loss_func == 'BCEWithLogitsLoss':
                loss = BCELogits_loss(cfg, device, targets, outputs, criterion, model, weights=None)
            elif cfg.loss_func == 'AUCM_MultiLabel':
                mask = ~torch.isnan(targets)
                mask_output = outputs[mask]
                mask_target = targets[mask]
                mask_output = mask_output.view(outputs.size(0), -1)
                mask_target = mask_target.view(targets.size(0), -1)
                mask_output = torch.sigmoid(mask_output)
                loss = criterion(mask_output, mask_target, auto=False)
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - Valid - Loss = {np.mean(avg_loss):4.4f}')
            #LOSS
            
            #AUC
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                    
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
        
        #AUC
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')
    avg_loss_ = np.mean(avg_loss)

    return auc, avg_loss_#, task_aucs, task_outputs, task_targets 