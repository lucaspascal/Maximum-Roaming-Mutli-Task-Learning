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
from min_norm_solvers import MinNormSolver, gradient_normalizers
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
        
        self.pred_tasks = nn.ModuleList()
        for k in range(14):
            self.pred_tasks.append(self.conv_layer([filter[0], 1], pred=True))
        self.pred_tasks.append(self.conv_layer([filter[0], 3], pred=True))


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

    
    def forward_shared(self, x):
        indices = [0] * 5
        for i in range(5):
            x = self.encoder_block[i](x)
            x, indices[i] = self.down_sampling(x)
        for i in range(5):
            x = self.up_sampling(x, indices[-i-1])
            x = self.decoder_block[i](x)
        
        return x

    
    def forward_task(self, x, t):
        t_pred = self.pred_tasks[t](x)
        if t < 13 :
            t_pred = F.sigmoid(t_pred)
        elif t==14 :
            t_pred = t_pred / torch.norm(t_pred, p=2, dim=1, keepdim=True)

        return t_pred

    
    def model_fit(self, x_pred, x_output, t):
        if t<13:
            # semantic loss: depth-wise cross entropy
            loss = torch.mean(F.binary_cross_entropy(x_pred, x_output, reduction='none'), dim=(0,2,3))
        elif t==13:
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
            # depth loss: l1 norm
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)
            loss = loss.unsqueeze(0)
        else:
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)
            # normal loss: dot product
            loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)
            loss = loss.unsqueeze(0)

        return loss

    def compute_task_miou(self, x_pred, x_output):
        batch_iou = 0.
        count = 0
        for i in range(x_pred.shape[0]):
            pred_mask = (x_pred[i,0,:,:]>=0.5).type(torch.float32)
            true_mask = x_output[i,0,:,:]
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
            batch_iou = torch.tensor(-1.)
        return batch_iou

    
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
num_tasks = 15 #MOO

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

        print('Epoch {}, Iter {}/{}'.format(epoch, k, nb_train_batches), end='\r')

        #################################################
        loss_data = {}
        scale = {}
        grads = {}
        task_losses = []
        ious = []
        pix_accs = []
        
        # Forward shared pass without gradients
        optimizer.zero_grad()
        with torch.no_grad():
            feats = model.forward_shared(train_data)
        rep_variable = Variable(feats.data.clone(), requires_grad=True)
        
        # Tasks forward passes with backprop on shared weights
        for t in range(num_tasks):
            optimizer.zero_grad()
            out_t = model.forward_task(rep_variable, t)
            ### scoring ###
            if t<13:
                label = train_label[:,t,None,:,:]
                ious.append(model.compute_task_miou(out_t, label).item())
                pix_accs.append(model.compute_pix_acc(out_t, label).item())
            elif t==13:
                label = train_depth
                abs_err, rel_err = model.depth_error(out_t, label)
            else:
                label = train_normal
                mean, med, map1, map2, map3 = model.normal_error(out_t, label)
            ### end scoring ###
            task_loss = model.model_fit(out_t, label, t)
            task_losses.append(task_loss[0].item())
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
        feats = model.forward_shared(train_data)
        for t in range(num_tasks):
            out_t = model.forward_task(feats, t)
            if t<13:
                label = train_label[:,t,None,:,:]
            elif t==13:
                label = train_depth
            else:
                label = train_normal
            loss_t = model.model_fit(out_t, label, t)
            loss_data[t] = loss_t.data
            if t > 0:
                loss = loss + scale[t]*loss_t
            else:
                loss = scale[t]*loss_t
        loss.backward()
        optimizer.step()

        # MGDA-UB end
        ##################################################
        # Scoring
        cost[0] = np.sum(task_losses[:13])
        cost[1] = np.mean(ious)
        cost[2] = np.mean(pix_accs)
        cost[3] = task_losses[-2]
        cost[4] = abs_err.item()
        cost[5] = rel_err.item()
        cost[6] = task_losses[-1]
        cost[7], cost[8], cost[9], cost[10], cost[11] = mean.item(), med.item(), map1.item(), map2.item(), map3.item()
        avg_cost[:12] += cost[:12] / nb_train_batches
        for i in range(len(ious)):
            if ious[i]>-1.:
                avg_cost[12+i] += ious[i]
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
                task_losses = []
                ious = []
                pix_accs = []
                feats = model.forward_shared(test_data)
                for t in range(num_tasks):
                    out_t = model.forward_task(feats, t)
                    ### scoring ###
                    if t<13:
                        label = test_label[:,t,None,:,:]
                        ious.append(model.compute_task_miou(out_t, label).item())
                        pix_accs.append(model.compute_pix_acc(out_t, label).item())
                    elif t==13:
                        label = test_depth
                        abs_err, rel_err = model.depth_error(out_t, label)
                    else:
                        label = test_normal
                        mean, med, map1, map2, map3 = model.normal_error(out_t, label)
                    task_loss = model.model_fit(out_t, label, t)
                    task_losses.append(task_loss[0].item())
                    ### end scoring ###

                
                # Scoring
                cost[25] = np.sum(task_losses[:13])
                cost[26] = np.mean(ious)
                cost[27] = np.mean(pix_accs)
                cost[28] = task_losses[-2]
                cost[29] = abs_err.item()
                cost[30] = rel_err.item()
                cost[31] = task_losses[-1]
                cost[32], cost[33], cost[34], cost[35], cost[36] = mean.item(), med.item(), map1.item(), map2.item(), map3.item()

                avg_cost[25:37] += cost[25:37] / nb_test_batches

                for i in range(len(ious)):
                    if ious[i]>-1.:
                        avg_cost[37+i] += ious[i]
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