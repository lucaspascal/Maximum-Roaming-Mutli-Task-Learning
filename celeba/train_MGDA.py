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
from min_norm_solvers import MinNormSolver, gradient_normalizers
from torch.autograd import Variable

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
model = resnet18(task_groups, trainable_weights=False).to(device)
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
        
        
        
    #################################################
        loss_data = {}
        scale = {}
        grads = {}
        preds = []
        class_losses = []
        task_losses = []
        
        # Forward shared pass without gradients
        optimizer.zero_grad()
        with torch.no_grad():
            feats = model.forward_shared(data)
        rep_variable = Variable(feats.data.clone(), requires_grad=True)
        
        # Tasks forward passes with backprop on shared weights
        for t in range(num_tasks):
            optimizer.zero_grad()
            out_t = torch.sigmoid(model.forward_task(rep_variable, t))
            preds.append((out_t>=0.5).type(torch.float32))
            class_loss = torch.mean(criterion(out_t, targets[t]), 0)
            task_loss = torch.mean(class_loss)
            class_losses.append(class_loss)
            task_losses.append(task_loss)
            loss_data[t] = task_loss.data
            task_loss.backward()
            grads[t] = [] 
            grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
            rep_variable.grad.data.zero_()
            
            
        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, 'none')
        for t in range(num_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(num_tasks)])
        for t in range(num_tasks):
            scale[t] = float(sol[t])            
            
            
            
        # Scaled back-propagation
        optimizer.zero_grad()
        feats = model.forward_shared(data)
        for t in range(num_tasks):
            out_t = torch.sigmoid(model.forward_task(feats, t))
            loss_t = torch.mean(criterion(out_t, targets[t]))
            loss_data[t] = loss_t.data
            if t > 0:
                loss = loss + scale[t]*loss_t
            else:
                loss = scale[t]*loss_t
        loss.backward()
        optimizer.step()

            
        ##################################################
        
        #class_losses = [torch.mean(criterion(logits[k], targets[k]), 0) for k in range(num_tasks)]
        #task_losses = torch.stack([torch.mean(elt) for elt in class_losses])
        #loss = torch.sum(model.weights*task_losses)
        
        
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
                 
