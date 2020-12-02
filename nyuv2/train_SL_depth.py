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
import math
import random
import utils
from dataset import NYUv2
from saver import Saver
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str, help='model name (for saving)')
parser.add_argument('--dataroot', default='datasets/nyuv2', type=str, help='dataset root')
parser.add_argument('--checkpoint_path', default='out', type=str, help='path for logs')
parser.add_argument('--recover', default=False, type=bool, help='whether or not to recover a checkpoint')
parser.add_argument('--reco_type', default='last_checkpoint', type=str, help='type of checkpoint to recover')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--eval_interval', default=5, type=int, help='interval between two evaluations')
parser.add_argument('--total_epoch', default=500, type=int, help='number of epochs')
opt = parser.parse_args()


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]

        self.class_nb = 13

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
        
        self.pred_task = self.conv_layer([filter[0], 1], pred=True)


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
                nn.BatchNorm2d(num_features=channel[1]),
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
        
        # define task prediction layers
        t_pred = self.pred_task(x)

        return t_pred

    
    def model_fit(self, x_pred, x_output):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)

        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)

        return loss

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)


##############################################################################
##############################################################################

################################     MAIN     ################################

##############################################################################
##############################################################################

# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)

# Define model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
model = SegNet().to(device)
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
metrics = ['DEPTH_LOSS', 'ABS_ERR', 'REL_ERR']
saver.add_metrics(metrics)

# Create datasets
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = opt.batch_size
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False)


# Define parameters
total_epoch = opt.total_epoch
nb_train_batches = len(nyuv2_train_loader)
nb_test_batches = len(nyuv2_test_loader)

# Iterations
while epoch < total_epoch:
    cost = np.zeros(2*len(metrics), dtype=np.float32)
    avg_cost = np.zeros(2*len(metrics), dtype=np.float32)


    ####################
    ##### Training #####
    ####################
    model.train()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in range(nb_train_batches):
        # Get data
        train_data, _, train_depth, _ = nyuv2_train_dataset.next()
        train_data, train_depth = train_data.to(device), train_depth.to(device)

        # Train step forward
        train_pred = model(train_data)
        loss = model.model_fit(train_pred, train_depth)
        print('Epoch {}, Iter {}/{}'.format(epoch, k, nb_train_batches), end='\r')

        # Train step backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scoring
        cost[0] = loss.item()
        cost[1], cost[2] = model.depth_error(train_pred, train_depth)
        avg_cost[:3] += cost[:3] / nb_train_batches

    ######################
    ##### Evaluation #####
    ######################
    if epoch%opt.eval_interval == 0 and epoch>=opt.eval_interval:
        print('EVALUATION')
        model.eval()
        with torch.no_grad():  # operations inside don't track history
            nyuv2_test_dataset = iter(nyuv2_test_loader)
            for k in range(nb_test_batches):
                # Get data
                test_data, _, test_depth, _ = nyuv2_test_dataset.next()
                test_data, test_depth = test_data.to(device),  test_depth.to(device)

                # Train step forward
                test_pred = model(test_data)
                loss = model.model_fit(test_pred, test_depth)

                # Scoring
                cost[3] = loss.item()
                cost[4], cost[5] = model.depth_error(test_pred, test_depth)

                avg_cost[3:6] += cost[3:6] / nb_test_batches

        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} '
              .format(epoch, avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3], avg_cost[4], avg_cost[5]))
        saver.save(model, epoch, avg_cost[:3], avg_cost[3:], optimizer=optimizer)

    epoch += 1