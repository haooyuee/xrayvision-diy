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
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
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

    #DIY optimizer
    if cfg.optimizer == None:
        raise ValueError('optimizer is not defined')
    else:
        #default optimizer
        if cfg.optimizer == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
        elif cfg.optimizer == 'PESG':
            optim = PESG(model, 
                 loss_fn=losses.AUCM_MultiLabel_V1(), 
                 lr=0.05, 
                 margin=1, 
                 epoch_decay=2e-3, 
                 weight_decay=1e-5)
        ##########################
        # can add more optimizer #
        ##########################
        else:
            raise ValueError('invalid optimizer')
    print('optim : ')
    print(optim)

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
            #raise ValueError('invalid loss function')
        elif cfg.loss_func == 'AUCM_MultiLabel':
            criterion = losses.AUCM_MultiLabel_V1(num_classes = 14)
        else:
            raise ValueError('invalid loss function')
    print('criterion : ')
    print(criterion)

 
    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(cfg.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
    
    for epoch in range(start_epoch, cfg.num_epochs):

        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               valid_loader=valid_loader,
                               criterion=criterion)
        
        auc_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion = torch.nn.CrossEntropyLoss()
                                     )[0]

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

    return metrics, best_metric, weights_for_best_validauc



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
        #print('targets + outputs')
        #print(targets)
        outputs = model(images)
        #print(outputs)


        if cfg.loss_func == 'BCEWithLogitsLoss':
            loss = BCELogits_loss(cfg, device, targets, outputs, criterion, model, weights)
        elif cfg.loss_func == 'AUCM_MultiLabel':
            mask = ~torch.isnan(targets)
            mask_output = outputs[mask]
            mask_target = targets[mask]
            #print('*********************************')
            #print(outputs)
            #print(targets)
            #print(mask)
            mask_output = mask_output.view(cfg.batch_size, -1)
            mask_target = mask_target.view(cfg.batch_size, -1)
            #print(mask_output)
            #print(mask_target)
            #raise ValueError('test')
            mask_output = torch.sigmoid(mask_output)
            loss = criterion(mask_output, mask_target)
            #print(loss)
    
        if cfg.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()
            
        if cfg.weightreg:
            loss += model.classifier.weight.abs().sum()
        

        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

        # validation
        if batch_idx % 100 == 0:
            model.eval()
            with torch.no_grad():    
                test_pred = []
                test_true = []
                limit = 0
                for jdx, samples in enumerate(valid_loader):

                    images = samples["img"].float().to(device)
                    targets = samples["lab"].to(device)
                    #test_data, test_label = data
                    #test_data = test_data.cuda()              
                    y_pred = model(images)

                    mask = ~torch.isnan(targets)
                    mask_y_pred = y_pred[mask]
                    mask_target = targets[mask]
                    test_pred.append(mask_y_pred.cpu().detach().numpy())
                    test_true.append(mask_target.cpu().detach().numpy())
                    #limit = limit + 1
                    #if limit >= 10:
                    #    break
                
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                val_auc =  np.mean(roc_auc_score(test_true, test_pred) )
                model.train()

                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                
            print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, batch_idx, val_auc, best_val_auc))
            if cfg.optimizer == 'PESG':
                print ('optimizer_lr=%.4f'%(optimizer.lr))

    return np.mean(avg_loss)

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=20000): #change limit
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
                    #print('+')
                    #print(loss)
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
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

    return auc, task_aucs, task_outputs, task_targets
