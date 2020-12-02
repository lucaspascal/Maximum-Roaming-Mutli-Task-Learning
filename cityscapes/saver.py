import os
import json
import numpy as np
import torch

class Saver():
    """
    Saving object for both metrics, weights and config of a model.
    """
    def __init__(self, path, args=None, metrics=None, iter=0):
        self.save_dir = path
        self.logs_dict = {'train':{}, 'val':{}}
        self.logs_file = os.path.join(self.save_dir,'logs.json')
        self.iter_nb = iter
        self.best_miou = 0.0
        self.best_rel_error = np.inf
        if args:
            self.config_dict = vars(args)
            self.config_dict['iter'] = self.iter_nb
            self.config_dict['best_miou'] = self.best_miou
            self.config_dict['best_rel_error'] = self.best_rel_error
        self.config_file = os.path.join(self.save_dir,'config.json')
        self.last_checkpoint_weights_file = os.path.join(self.save_dir,'last_checkpoint_weights.pth')
        self.best_miou_weights_file = os.path.join(self.save_dir,'best_miou_weights.pth')
        self.best_rel_error_weights_file = os.path.join(self.save_dir,'best_rel_error_weights.pth')
        self.metrics = [] if metrics==None else metrics
        
    def relocate(self, new_path):
        self.save_dir = new_path
        self.logs_dict = {'train':{}, 'val':{}}
        self.logs_file = os.path.join(self.save_dir,'logs.json')
        self.iter_nb = 100
        self.best_miou = 0.0
        self.best_rel_error = np.inf
        self.config_dict['iter'] = self.iter_nb
        self.config_dict['best_miou'] = self.best_miou
        self.config_dict['best_rel_error'] = self.best_rel_error
        self.config_file = os.path.join(self.save_dir,'config.json')
        self.last_checkpoint_weights_file = os.path.join(self.save_dir,'last_checkpoint_weights.pth')
        self.best_miou_weights_file = os.path.join(self.save_dir,'best_miou_weights.pth')
        self.best_rel_error_weights_file = os.path.join(self.save_dir,'best_rel_error_weights.pth')
        
    def add_metrics(self, names):
        self.metrics += names

    def quick_save(self, model, save_name, optimizer=None):
        save_path = os.path.join(self.save_dir, save_name+'_weights.pth')
        if optimizer:
            opt_weights = optimizer.get_weights()
            np.save(os.path.join(self.save_dir,save_name+'_opt_weights'), opt_weights)
        model.save_weights(save_path, save_format='h5')
        
        
    def save(self, model, iter_nb, train_metrics_values, test_metrics_values, tasks_weights=[], optimizer=None):
        self.logs_dict['train'][str(iter_nb)] = {}
        self.logs_dict['val'][str(iter_nb)] = {}
        for k in range(len(self.metrics)):
            self.logs_dict['train'][str(iter_nb)][self.metrics[k]] = float(train_metrics_values[k])
            self.logs_dict['val'][str(iter_nb)][self.metrics[k]] = float(test_metrics_values[k])
        
        if len(tasks_weights)>0 :
            for k in range(len(tasks_weights)):
                self.logs_dict['val'][str(iter_nb)]['weight_'+str(k)] = tasks_weights[k]

        with open(self.logs_file, 'w') as f:
            json.dump(self.logs_dict, f)  
            
        ckpt = {
                'model_state_dict': model.state_dict(),
                'iter_nb': iter_nb,
                }
        if optimizer:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()


        # Saves best miou score if reached
        if 'MEAN_IOU' in self.metrics:
            miou = float(test_metrics_values[self.metrics.index('MEAN_IOU')])
            if miou > self.best_miou and iter_nb>0:
                print('Best miou. Saving it.')
                torch.save(ckpt, self.best_miou_weights_file)
                self.best_miou = miou
                self.config_dict['best_miou'] = self.best_miou
        # Saves best relative error if reached
        if 'REL_ERR' in self.metrics:
            rel_error = float(test_metrics_values[self.metrics.index('REL_ERR')])
            if rel_error < self.best_rel_error and iter_nb>0:
                print('Best rel error. Saving it.')
                torch.save(ckpt, self.best_rel_error_weights_file)
                self.best_rel_error = rel_error
                self.config_dict['best_rel_error'] = self.best_rel_error
            
        # Saves last checkpoint
        torch.save(ckpt, self.last_checkpoint_weights_file)
        self.iter_nb = iter_nb
        self.config_dict['iter'] = self.iter_nb
        with open(self.config_file, 'w') as f:
            json.dump(self.config_dict, f)
                
                
    def load(self):
        if os.path.isfile(self.logs_file):
            with open(self.logs_file) as f:
                self.logs_dict = json.load(f)
        if os.path.isfile(self.config_file):
            with open(self.config_file) as f:
                prev_config = json.load(f)
                self.iter_nb = prev_config['iter']
                self.best_miou = prev_config['best_miou'] if 'best_miou' in prev_config.keys() else 0.
                self.best_rel_error = prev_config['best_rel_error'] if 'best_rel_error' in prev_config.keys() else np.inf
    
    