import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
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
        
        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)


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
        t1_pred = F.sigmoid(self.pred_task1(x))
        t2_pred = self.pred_task2(x)
        t3_pred = self.pred_task3(x)
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred]

    
    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3, x_output3):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)

        # semantic loss: depth-wise cross entropy
        #loss1 = F.binary_cross_entropy(x_pred1, x_output1)
        loss1 = torch.mean(F.binary_cross_entropy(x_pred1, x_output1, reduction='none'), dim=(0,2,3))

        # depth loss: l1 norm
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)

        # normal loss: dot product
        loss3 = 1 - torch.sum((x_pred3 * x_output3) * binary_mask) / torch.nonzero(binary_mask).size(0)

        return torch.cat((loss1, loss2.unsqueeze(0), loss3.unsqueeze(0)))

    def compute_miou(self, x_pred, x_output):
        batch_miou = 0.
        batch_ious = torch.zeros(13, dtype=torch.float32)
        counts = torch.zeros(13)
        for i in range(x_pred.shape[0]):
            pred_mask = (x_pred[i,:,:,:]>=0.5).type(torch.float32)
            true_mask = x_output[i,:,:,:]
            mask_comb = pred_mask + true_mask
            union = torch.sum((mask_comb > 0).type(torch.FloatTensor), (1,2))
            intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor), (1,2))
            
            ious = -torch.ones(union.shape, dtype=torch.float32)
            for j in range(union.shape[0]):
                if union[j] != 0:
                    ious[j] = intsec[j]/union[j]
            miou = torch.mean(ious[ious!=-1])
            batch_miou += miou
            batch_ious[ious!=-1] += ious[ious!=-1]
            counts[ious!=-1] += 1
        batch_miou /= x_pred.shape[0]
        batch_ious[counts>0] /= counts[counts>0]
        batch_ious[counts==0] = -1
        return batch_miou, batch_ious
                

    def compute_pix_acc(self, x_pred, x_output):
        pred_mask = (x_pred>=0.5).type(torch.float32)
        true_mask = x_output
        pixel_acc = torch.mean(torch.eq(pred_mask,true_mask).type(torch.float32))

        return pixel_acc
            
    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


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
metrics = ['SEMANTIC_LOSS', 'MEAN_IOU', 'PIX_ACC', 'DEPTH_LOSS', 'ABS_ERR', 'REL_ERR', 'NORMAL_LOSS', 'MEAN', 'MED', '<11.25', '<22.5', '<30', 'IOU_0', 'IOU_1', 'IOU_2', 'IOU_3', 'IOU_4', 'IOU_5', 'IOU_6', 'IOU_7', 'IOU_8', 'IOU_9', 'IOU_10', 'IOU_11', 'IOU_12']
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
    ious_counts = np.zeros(13)
    for k in range(nb_train_batches):
        # Get data
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_depth, train_normal = train_depth.to(device), train_normal.to(device)

        # Train step forward
        train_pred = model(train_data)
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)
        loss = torch.sum(train_loss)
        print('Epoch {}, Iter {}/{}'.format(epoch, k, nb_train_batches), end='\r')

        # Train step backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scoring
        miou, ious = model.compute_miou(train_pred[0], train_label)
        np_ious = ious.numpy()
        cost[0] = torch.sum(train_loss[:13]).item()
        cost[1] = miou.item()
        cost[2] = model.compute_pix_acc(train_pred[0], train_label).item()
        cost[3] = train_loss[-2].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[-1].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
        avg_cost[:12] += cost[:12] / nb_train_batches
        for i in range(len(np_ious)):
            if np_ious[i]>-1.:
                avg_cost[12+i] += np_ious[i]
                ious_counts[i] += 1

    for k in range(13):
        if ious_counts[k] != 0:
            avg_cost[12+k] /= ious_counts[k]
    avg_cost[1] = np.mean(avg_cost[12:25])

    ######################
    ##### Evaluation #####
    ######################
    if epoch%opt.eval_interval == 0 and epoch>=opt.eval_interval:
        print('EVALUATION')
        model.eval()
        with torch.no_grad():  # operations inside don't track history
            nyuv2_test_dataset = iter(nyuv2_test_loader)
            ious_counts = np.zeros(13)
            for k in range(nb_test_batches):
                # Get data
                test_data, test_label, test_depth, test_normal = nyuv2_test_dataset.next()
                test_data, test_label = test_data.to(device),  test_label.to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                # Train step forward
                test_pred = model(test_data)
                test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

                # Scoring
                miou, ious = model.compute_miou(test_pred[0], test_label)
                np_ious = ious.numpy()
                cost[25] = torch.sum(test_loss[:13]).item()
                cost[26] = miou.item()
                cost[27] = model.compute_pix_acc(test_pred[0], test_label).item()
                cost[28] = test_loss[-2].item()
                cost[29], cost[30] = model.depth_error(test_pred[1], test_depth)
                cost[31] = test_loss[-1].item()
                cost[32], cost[33], cost[34], cost[35], cost[36] = model.normal_error(test_pred[2], test_normal)

                avg_cost[25:37] += cost[25:37] / nb_test_batches

                for i in range(len(np_ious)):
                    if np_ious[i]>-1.:
                        avg_cost[37+i] += np_ious[i]
                        ious_counts[i] += 1

            for k in range(13):
                if ious_counts[k] != 0:
                    avg_cost[37+k] /= ious_counts[k]
            avg_cost[26] = np.mean(avg_cost[37:50])
    
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              .format(epoch, avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3],
                    avg_cost[4], avg_cost[5], avg_cost[6], avg_cost[7], avg_cost[8], avg_cost[9],
                    avg_cost[10], avg_cost[11], avg_cost[25], avg_cost[26],
                    avg_cost[27], avg_cost[28], avg_cost[29], avg_cost[30], avg_cost[31],
                    avg_cost[32], avg_cost[33], avg_cost[34], avg_cost[35], avg_cost[36]))
        saver.save(model, epoch, avg_cost[:25], avg_cost[25:], optimizer=optimizer)

    epoch += 1