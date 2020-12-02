import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from MR_layer import MR
import time

import utils
from dataset import Cityscapes
from saver import Saver
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str, help='model name (for saving)')
parser.add_argument('--dataroot', default='datasets/cityscapes', type=str, help='dataset root')
parser.add_argument('--checkpoint_path', default='out', type=str, help='path for logs')
parser.add_argument('--recover', default=False, type=bool, help='whether or not to recover a checkpoint')
parser.add_argument('--reco_type', default='last_checkpoint', type=str, help='type of checkpoint to recover')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--eval_interval', default=5, type=int, help='interval between two evaluations')
parser.add_argument('--total_epoch', default=500, type=int, help='number of epochs')
parser.add_argument('--sigma', default=0.6, type=float, help='sharing ratio')
opt = parser.parse_args()


class SegNet(nn.Module):
    def change_task(self, task):
        def aux(m):
            if hasattr(m, 'active_task'):
                m.set_active_task(task)
        self.apply(aux)

    def set_active_task(self, active_task):
        self.active_task = active_task
    
    def get_block(self, depth):
        if depth < len(self.enc_struct):
            seq_nb, elt_nb = self.enc_struct[depth]
            conv_block = self.encoder_block[seq_nb][elt_nb]
        else:
            seq_nb, elt_nb = self.dec_struct[depth-len(self.enc_struct)]
            conv_block = self.decoder_block[seq_nb][elt_nb]
        return conv_block
    
    def get_weights(self, depth):
        conv_block = self.get_block(depth)
        weights = conv_block[0].weight
        return weights
    
    def get_routing_mask(self, depth):
        conv_block = self.get_block(depth)
        TR_layer = conv_block[2]
        mapping = TR_layer.unit_mapping
        tested_tasks = TR_layer.tested_tasks
        return mapping.detach().cpu().numpy(), tested_tasks
    
    def __init__(self, sigma=0.4, recover=False, reco_type='last_checkpoint', path=None, optimizer=None):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 7
        self.sigma = sigma
        self.active_task = 0
        self.enc_struct = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (4,0), (4,1), (4,2)]
        self.dec_struct = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0), (3,1), (4,0), (4,1)]
        
        self.encoder_block = nn.ModuleList([nn.Sequential(self.conv_layer([3, filter[0]]), 
                                                          self.conv_layer([filter[0], filter[0]])),
                                            nn.Sequential(self.conv_layer([filter[0], filter[1]]), 
                                                          self.conv_layer([filter[1], filter[1]])),
                                            nn.Sequential(self.conv_layer([filter[1], filter[2]]), 
                                                          self.conv_layer([filter[2], filter[2]]),
                                                          self.conv_layer([filter[2], filter[2]])),
                                            nn.Sequential(self.conv_layer([filter[2], filter[3]]), 
                                                          self.conv_layer([filter[3], filter[3]]),
                                                          self.conv_layer([filter[3], filter[3]])),
                                            nn.Sequential(self.conv_layer([filter[3], filter[4]]), 
                                                          self.conv_layer([filter[4], filter[4]]),
                                                          self.conv_layer([filter[4], filter[4]])),
                                           ])
        
        self.decoder_block = nn.ModuleList([nn.Sequential(self.conv_layer([filter[4], filter[3]]), 
                                                          self.conv_layer([filter[3], filter[3]]),
                                                          self.conv_layer([filter[3], filter[3]])),
                                            nn.Sequential(self.conv_layer([filter[3], filter[2]]), 
                                                          self.conv_layer([filter[2], filter[2]]),
                                                          self.conv_layer([filter[2], filter[2]])),
                                            nn.Sequential(self.conv_layer([filter[2], filter[1]]), 
                                                          self.conv_layer([filter[1], filter[1]]),
                                                          self.conv_layer([filter[1], filter[1]])),
                                            nn.Sequential(self.conv_layer([filter[1], filter[0]]), 
                                                          self.conv_layer([filter[0], filter[0]])),
                                            nn.Sequential(self.conv_layer([filter[0], filter[0]]), 
                                                          self.conv_layer([filter[0], filter[0]])),
                                           ])
        
        self.pred_tasks = nn.ModuleList()
        for k in range(8):
            self.pred_tasks.append(self.conv_layer([filter[0], 1], pred=True))


        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                


    # define convolutional block
    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1], track_running_stats=False),
                MR(channel[1], 8, self.sigma, self.active_task),
                nn.ReLU(inplace=True)
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def forward(self, x):
        indices = [0] * 5
        for i in range(5):
            x = self.encoder_block[i](x)
            x, indices[i] = self.down_sampling(x)
        for i in range(5):
            x = self.up_sampling(x, indices[-i-1])
            x = self.decoder_block[i](x)
        
        t_pred = self.pred_tasks[self.active_task](x)
        if self.active_task < 7:
            t_pred = F.sigmoid(t_pred)

        return t_pred
    

    def model_fit(self, x_pred, x_output1, x_output2):
        if self.active_task <7 :
            # semantic loss: depth-wise cross entropy
            loss = F.binary_cross_entropy(x_pred, x_output1[:, self.active_task,None,:,:])

        else:
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
            # depth loss: l1 norm
            loss = torch.sum(torch.abs(x_pred - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)

        return loss

    
    def compute_miou(self, x_pred, x_output):
        batch_iou = 0.
        count = 0
        for i in range(x_pred.shape[0]):
            pred_mask = (x_pred[i,0,:,:]>=0.5).type(torch.float32)
            true_mask = x_output[i,self.active_task,:,:]
            mask_comb = pred_mask + true_mask
            union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            
            ious = -torch.ones(union.shape, dtype=torch.float32)
            if union == 0:
                continue
            count += 1
            batch_iou += intsec/union
        if count>0:
            batch_iou /= count
        else:
            batch_iou = 0.
            print('DIV BY 0')
        return batch_iou
    
    
    
    def compute_pix_acc(self, x_pred, x_output):
        pred_mask = (x_pred>=0.5).type(torch.float32)
        true_mask = x_output[:,self.active_task,:,:].unsqueeze(1)
        pixel_acc = torch.mean(torch.eq(pred_mask,true_mask).type(torch.float32))

        return pixel_acc

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)


##################################################################
##################################################################

##########################     MAIN     ##########################

##################################################################
##################################################################

# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)

# Define model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
model = SegNet(sigma=opt.sigma).to(device)
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
metrics = ['SEMANTIC_LOSS', 'IOU_0', 'IOU_1', 'IOU_2', 'IOU_3', 'IOU_4', 'IOU_5', 'IOU_6', 'MEAN_IOU', 'PIX_ACC', 'DEPTH_LOSS', 'ABS_ERR', 'REL_ERR']
saver.add_metrics(metrics)

# Create datasets
dataset_path = opt.dataroot
city_train_set = Cityscapes(root=dataset_path, train=True)
city_test_set = Cityscapes(root=dataset_path, train=False)

batch_size = opt.batch_size
city_train_loader = torch.utils.data.DataLoader(
    dataset=city_train_set,
    batch_size=batch_size,
    shuffle=True)

city_test_loader = torch.utils.data.DataLoader(
    dataset=city_test_set,
    batch_size=batch_size,
    shuffle=False)


# Define parameters
total_epoch = opt.total_epoch
nb_train_batches = len(city_train_loader)
nb_test_batches = len(city_test_loader)
lambda_weight = np.ones(8)

# Iterations
while epoch < total_epoch:
    cost = np.zeros(2*len(metrics), dtype=np.float32)
    avg_cost = np.zeros(2*len(metrics), dtype=np.float32)
    
    ####################
    ##### Training #####
    ####################
    model.train()
    city_train_dataset = iter(city_train_loader)
    for k in range(nb_train_batches):
        # Get data
        train_data, train_label, train_depth = city_train_dataset.next()
        train_data, train_label, train_depth = train_data.to(device), train_label.to(device), train_depth.to(device)
        
        # Train step forward
        optimizer.zero_grad()

        for task in range(8):
            model.change_task(task)
            train_pred = model(train_data)
            train_loss = model.model_fit(train_pred, train_label, train_depth)
            loss = torch.mean(lambda_weight[task] * train_loss)
            loss.backward()
            print('Epoch {}, Iter {}/{}, Active task {}'.format(epoch, k, nb_train_batches, task+1), end='\r')
        
            # Scoring
            if task < 7:
                avg_cost[0] += (train_loss.item()/7) / nb_train_batches
                avg_cost[1+task] += (model.compute_miou(train_pred, train_label).item()) / nb_train_batches
                avg_cost[8] = np.mean(avg_cost[1:8])
                avg_cost[9] += (model.compute_pix_acc(train_pred, train_label).item()/7) / nb_train_batches
            else :
                avg_cost[10] += (train_loss.item()) / nb_train_batches
                abs_err, rel_err = model.depth_error(train_pred, train_depth)
                avg_cost[11] += abs_err / nb_train_batches
                avg_cost[12] += rel_err / nb_train_batches

        # Train step backward
        optimizer.step()

    
    ######################
    ##### Evaluation #####
    ######################
    if epoch%opt.eval_interval == 0 and epoch>=opt.eval_interval:
        print('EVALUATION')
        model.eval()
        with torch.no_grad():  # operations inside don't track history
            city_test_dataset = iter(city_test_loader)
            for k in range(nb_test_batches):
                # Get data
                test_data, test_label, test_depth = city_test_dataset.next()
                test_data, test_label, test_depth = test_data.to(device),  test_label.to(device), test_depth.to(device)


                # Loop forward pass and scoring over each task
                for task in range(8):
                    print('Evaluation, Iter {}/{}, Task {}'.format(k, nb_test_batches, task), end='\r')
                    model.change_task(task)
                    test_pred = model(test_data)
                    test_loss = model.model_fit(test_pred, test_label, test_depth)

                    # Scoring
                    if task < 7:
                        avg_cost[13] += (test_loss.item()/7) / nb_test_batches
                        avg_cost[14+task] += (model.compute_miou(test_pred, test_label).item()) / nb_test_batches
                        avg_cost[21] = np.mean(avg_cost[14:21])
                        avg_cost[22] += (model.compute_pix_acc(test_pred, test_label).item()/7) / nb_test_batches
                    else :
                        avg_cost[23] += (test_loss.item()) / nb_test_batches
                        abs_err, rel_err = model.depth_error(test_pred, test_depth)
                        avg_cost[24] += abs_err / nb_test_batches
                        avg_cost[25] += rel_err / nb_test_batches

                
        # Loggin and saving
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '.format(epoch, avg_cost[0], avg_cost[8], avg_cost[9], avg_cost[10], avg_cost[11], avg_cost[12], avg_cost[13], avg_cost[21], avg_cost[22], avg_cost[23], avg_cost[24], avg_cost[25]))
        
        saver.save(model, epoch, avg_cost[:len(metrics)], avg_cost[len(metrics):], optimizer=optimizer)
    
    epoch += 1