import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
import utils
from dataset import CelebaGroupedDataset
from resnet_MTL import resnet18
from saver import Saver
import random
import math

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str)
parser.add_argument('--dataroot', default='datasets/celeba/', type=str)
parser.add_argument('--checkpoint_path', default='out')
parser.add_argument('--recover', default=False, type=bool)
parser.add_argument('--reco_type', default='last_checkpoint', type=str)
parser.add_argument('--total_epoch', default=40, type=int)
parser.add_argument('--image_size', default=64, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
opt = parser.parse_args()



task_groups = [
               [2,10,13,14,18,20,25,26,39],
               [3,15,23,1,12],
               [4,5,8,9,11,17,28,32,33],
               [6,21,31,36],
               [7,27],
               [0,16,22,24,30],
               [19,29],
               [34,35,37,38],
              ]
    
num_tasks = len(task_groups)
num_classes = sum([len(elt) for elt in task_groups])


# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)


# Define device, model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
model = resnet18(task_groups, trainable_weights=True).to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Recover weights, if required
if opt.recover:
    ckpt_file = os.path.join(model_dir, opt.reco_type+'_weights.pth')
    ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt['iter_nb'] + 1
    print('Model recovered from {}.'.format(ckpt_file))
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print('Optimizer recovered from {}.'.format(ckpt_file))
    saver.load()
else:
    epoch=0


# Metrics
metrics = ['LOSS', 'ACC', 'PREC', 'REC', 'FSCORE']
metrics += ['LOSS_{}'.format(c) for c in range(num_classes)]
metrics += ['ACC_{}'.format(c) for c in range(num_classes)]
metrics += ['PREC_{}'.format(c) for c in range(num_classes)]
metrics += ['REC_{}'.format(c) for c in range(num_classes)]
metrics += ['FSCORE_{}'.format(c) for c in range(num_classes)]
saver.add_metrics(metrics)

# Create datasets
train_set = CelebaGroupedDataset(opt.dataroot,
                                 task_groups,
                                 split='train',
                                 image_size=opt.image_size)
val_set = CelebaGroupedDataset(opt.dataroot,
                               task_groups,
                               split='val',
                               image_size=opt.image_size)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=6)
val_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=6)

# Set different values and meters
train_batches = len(train_loader)
val_batches = len(val_loader)
nb_iter = 0


# Iterations
while epoch<opt.total_epoch:
    # Train loop
    model.train()
    train_dataset = iter(train_loader)
    train_losses = torch.zeros(num_classes, dtype=torch.float32)
    train_well_pred = torch.zeros(num_classes, dtype=torch.float32)
    train_to_pred = torch.zeros(num_classes, dtype=torch.float32)
    train_pred = torch.zeros(num_classes, dtype=torch.float32)
    train_accs = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(train_batches):
        # Get data
        data, targets = train_dataset.next()
        data, targets = data.to(device), [elt.to(device) for elt in targets]
        
        feats = model(data)
        logits = [torch.sigmoid(elt) for elt in feats]
        preds = [(elt>=0.5).type(torch.float32) for elt in logits]
        class_losses = [torch.mean(criterion(logits[k], targets[k]), 0) for k in range(num_tasks)]
        task_losses = torch.stack([torch.mean(elt) for elt in class_losses])
        
        loss = torch.sum(model.weights*task_losses)
        
        # Backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        
        ################################
        # zero the w_i(t) gradients 
        model.weights.grad = 0.0 * model.weights.grad
        W = model.get_weights()[-1]
        norms = []
        
        for w_i, L_i in zip(model.weights, task_losses):
            # gradient of L_i(t) w.r.t. W
            gLgW = torch.autograd.grad(L_i, W, retain_graph=True)
            
            # G^{(i)}_W(t)
            norms.append(torch.norm(w_i * gLgW[0]))
            
        norms = torch.stack(norms)
        
        # set L(0)
        # if using log(C) init, remove these two lines
        if i == 0:
            initial_losses = task_losses.detach()
        
        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
            
            # loss ratios \curl{L}(t)
            loss_ratios = task_losses / initial_losses
            
            # inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = norms.mean() * (inverse_train_rates ** opt.alpha)
        
        # write out the gradnorm loss L_grad and set the weight gradients
        grad_norm_loss = (norms - constant_term).abs().sum()
        model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
        
        # apply gradient descent
        optimizer.step()
        
        # renormalize the gradient weights
        with torch.no_grad():
            
            normalize_coeff = len(model.weights) / model.weights.sum()
            model.weights.data = model.weights.data * normalize_coeff
        
        # GRADNORM END
        ################################
        
        
        
        # Scoring
        with torch.no_grad():
            for task in range(num_tasks):
                train_losses[task_groups[task]] += class_losses[task].cpu() / train_batches
                train_pred[task_groups[task]] += torch.sum(preds[task].cpu(), dim=0)
                train_to_pred[task_groups[task]] += torch.sum(targets[task].cpu(), dim=0)
                train_well_pred[task_groups[task]] += torch.sum((preds[task]*targets[task]).cpu(), dim=0)
                train_accs[task_groups[task]] += torch.mean((preds[task]==targets[task]).cpu().type(torch.float32), axis=0)/train_batches

                
        # Incr iter
        nb_iter += 1
        
        # Avg scores
        train_precs = train_well_pred / (train_pred + 1e-7)
        train_recs = train_well_pred / (train_to_pred + 1e-7)
        train_fscores = 2*train_precs*train_recs/(train_precs+train_recs+1e-7)
        

        # Out line
        print('Epoch {}, iter {}/{}, Loss : {}'.format(epoch, i+1, train_batches, loss.item()), end='\r')
        
    #############
    # Eval loop #
    #############
    model.eval()
    with torch.no_grad(): 
        val_dataset = iter(val_loader)
        val_losses = torch.zeros(num_classes, dtype=torch.float32)
        val_well_pred = torch.zeros(num_classes, dtype=torch.float32)
        val_to_pred = torch.zeros(num_classes, dtype=torch.float32)
        val_pred = torch.zeros(num_classes, dtype=torch.float32)
        val_accs = torch.zeros(num_classes, dtype=torch.float32)
        for i in range(val_batches):
            print('Eval iter {}/{}'.format(i+1, val_batches), end='\r')
                  
            # Get data
            data, targets = val_dataset.next()
            data, targets = data.to(device), [elt.to(device) for elt in targets]

            # Forward
            feats = model(data)
            logits = [torch.sigmoid(elt) for elt in feats]
            preds = [(elt>=0.5).type(torch.float32) for elt in logits]
            class_losses = [torch.mean(criterion(logits[k], targets[k]), 0) for k in range(num_tasks)]
            task_losses = [torch.mean(elt) for elt in class_losses]
        
        
            # Scoring
            for task in range(num_tasks):
                val_losses[task_groups[task]] += class_losses[task].cpu() / val_batches
                val_pred[task_groups[task]] += torch.sum(preds[task].cpu(), dim=0)
                val_to_pred[task_groups[task]] += torch.sum(targets[task].cpu(), dim=0)
                val_well_pred[task_groups[task]] += torch.sum((preds[task]*targets[task]).cpu(), dim=0)
                val_accs[task_groups[task]] += torch.mean((preds[task]==targets[task]).cpu().type(torch.float32), axis=0)/val_batches
                
                
        # Avg scores
        val_precs = val_well_pred / (val_pred + 1e-7)
        val_recs = val_well_pred / (val_to_pred + 1e-7)
        val_fscores = 2*val_precs*val_recs/(val_precs+val_recs+1e-7)
        

        # Out line
        print('EVAL EPOCH {}, Loss : {}, acc : {}, prec : {}, rec : {}, f : {}'.format(epoch, torch.sum(val_losses).item(), torch.mean(val_accs).item(), torch.mean(val_precs).item(), torch.mean(val_recs).item(), torch.mean(val_fscores).item()))
        
        # Saving
        train_metrics = [torch.sum(train_losses).item(), torch.mean(train_accs).item(), torch.mean(train_precs).item(), torch.mean(train_recs).item(), torch.mean(train_fscores).item()] + train_losses.numpy().tolist() + train_accs.numpy().tolist() + train_precs.numpy().tolist() + train_recs.numpy().tolist() + train_fscores.numpy().tolist()
        val_metrics = [torch.sum(val_losses).item(), torch.mean(val_accs).item(), torch.mean(val_precs).item(), torch.mean(val_recs).item(), torch.mean(val_fscores).item()] + val_losses.numpy().tolist() + val_accs.numpy().tolist() + val_precs.numpy().tolist() + val_recs.numpy().tolist() + val_fscores.numpy().tolist()
    
        saver.save(model, epoch, train_metrics, val_metrics, optimizer=optimizer)
    
    epoch += 1
                 
                 
