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
        self.best_fscore = 0.0
        if args:
            self.config_dict = vars(args)
            self.config_dict['iter'] = self.iter_nb
            self.config_dict['best_fscore'] = self.best_fscore
        self.config_file = os.path.join(self.save_dir,'config.json')
        self.last_checkpoint_weights_file = os.path.join(self.save_dir,'last_checkpoint_weights.pth')
        self.best_fscore_weights_file = os.path.join(self.save_dir,'best_fscore_weights.pth')
        self.metrics = [] if metrics==None else metrics
        
    def add_metrics(self, names):
        self.metrics += names

        
    def save(self, model, iter_nb, train_metrics_values, test_metrics_values, optimizer=None):
        self.logs_dict['train'][str(iter_nb)] = {}
        self.logs_dict['val'][str(iter_nb)] = {}
        for k in range(len(self.metrics)):
            self.logs_dict['train'][str(iter_nb)][self.metrics[k]] = float(train_metrics_values[k])
            self.logs_dict['val'][str(iter_nb)][self.metrics[k]] = float(test_metrics_values[k])
        

        with open(self.logs_file, 'w') as f:
            json.dump(self.logs_dict, f)  
            
        ckpt = {
                'model_state_dict': model.state_dict(),
                'iter_nb': iter_nb,
                }
        if optimizer:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()


        fscore = float(test_metrics_values[self.metrics.index('FSCORE')])
        # Saves best miou score if reached
        if fscore > self.best_fscore :
            print('Best fscore. Saving it.')
            torch.save(ckpt, self.best_fscore_weights_file)
            self.best_fscore = fscore
            self.config_dict['best_fscore'] = self.best_fscore
            
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
                self.best_fscore = prev_config['best_fscore'] if 'best_fscore' in prev_config.keys() else 0.
    
    