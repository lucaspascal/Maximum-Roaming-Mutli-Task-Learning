import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
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
parser.add_argument('--alpha', default=0.5, type=float)
opt = parser.parse_args()


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]

        self.class_nb = 7
        self.weights = torch.nn.Parameter(torch.ones(8, dtype=torch.float32), requires_grad=True)
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
        
        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)


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


                
                
    def get_block(self, depth):
        if depth < len(self.enc_struct):
            seq_nb, elt_nb = self.enc_struct[depth]
            conv_block = self.encoder_block[seq_nb][elt_nb]
        else:
            seq_nb, elt_nb = self.dec_struct[depth-len(self.enc_struct)]
            conv_block = self.decoder_block[seq_nb][elt_nb]
        return conv_block
    
    def get_weight(self, depth):
        conv_block = self.get_block(depth)
        weights = conv_block[0].weight
        return weights
    
    def get_weights(self):
        return [self.get_weight(depth) for depth in range(26)]
    
    
    
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

        return [t1_pred, t2_pred]

    
    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)

        # semantic loss: depth-wise cross entropy
        loss1 = torch.mean(F.binary_cross_entropy(x_pred1, x_output1, reduction='none'), dim=(0,2,3))

        # depth loss: l1 norm
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)

        return torch.cat((loss1, loss2.unsqueeze(0)))

    def compute_miou(self, x_pred, x_output):
        batch_miou = 0.
        batch_ious = torch.zeros(7, dtype=torch.float32)
        counts = torch.zeros(7)
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
                

    def compute_iou(self, x_pred, x_output):
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
metrics = ['SEMANTIC_LOSS', 'MEAN_IOU', 'PIX_ACC', 'DEPTH_LOSS', 'ABS_ERR', 'REL_ERR', 'IOU_0', 'IOU_1', 'IOU_2', 'IOU_3', 'IOU_4', 'IOU_5', 'IOU_6']
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

# Iterations
while epoch < total_epoch:
    cost = np.zeros(2*len(metrics), dtype=np.float32)
    avg_cost = np.zeros(2*len(metrics), dtype=np.float32)


    ####################
    ##### Training #####
    ####################
    model.train()
    city_train_dataset = iter(city_train_loader)
    ious_counts = np.zeros(7)
    for k in range(nb_train_batches):
        # Get data
        train_data, train_label, train_depth = city_train_dataset.next()
        train_data, train_label, train_depth = train_data.to(device), train_label.to(device), train_depth.to(device)

        # Train step forward
        train_pred = model(train_data)
        task_losses = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth)
        loss = torch.sum(model.weights*task_losses)
        print('Epoch {}, Iter {}/{}'.format(epoch, k, nb_train_batches), end='\r')

        # Train step backward
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
        if k == 0:
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
        miou, ious = model.compute_miou(train_pred[0], train_label)
        np_ious = ious.numpy()
        cost[0] = torch.sum(task_losses[:7]).item()
        cost[1] = miou.item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = task_losses[-1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        avg_cost[:6] += cost[:6] / nb_train_batches
        for i in range(len(np_ious)):
            if np_ious[i]>-1.:
                avg_cost[6+i] += np_ious[i]
                ious_counts[i] += 1

    avg_cost[6:13] /= ious_counts
    
    ######################
    ##### Evaluation #####
    ######################
    if epoch%opt.eval_interval == 0 and epoch>=opt.eval_interval:
        print('EVALUATION')
        model.eval()
        with torch.no_grad():  # operations inside don't track history
            city_test_dataset = iter(city_test_loader)
            ious_counts = np.zeros(7)
            for k in range(nb_test_batches):
                # Get data
                test_data, test_label, test_depth = city_test_dataset.next()
                test_data, test_label, test_depth = test_data.to(device),  test_label.to(device), test_depth.to(device)

                # Train step forward
                test_pred = model(test_data)
                test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth)

                # Scoring
                miou, ious = model.compute_miou(test_pred[0], test_label)
                np_ious = ious.numpy()
                cost[13] = torch.sum(test_loss[:7]).item()
                cost[14] = miou.item()
                cost[15] = model.compute_iou(test_pred[0], test_label).item()
                cost[16] = test_loss[-1].item()
                cost[17], cost[18] = model.depth_error(test_pred[1], test_depth)

                avg_cost[13:19] += cost[13:19] / nb_test_batches

                for i in range(len(np_ious)):
                    if np_ious[i]>-1.:
                        avg_cost[19+i] += np_ious[i]
                        ious_counts[i] += 1

            avg_cost[19:26] /= ious_counts
    
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
              .format(epoch, avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3], avg_cost[4], avg_cost[5], avg_cost[13], avg_cost[14], avg_cost[15], avg_cost[16], avg_cost[17], avg_cost[18]))
        saver.save(model, epoch, avg_cost[:13], avg_cost[13:], optimizer=optimizer)

    epoch += 1