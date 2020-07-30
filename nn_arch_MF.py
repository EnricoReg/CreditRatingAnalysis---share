# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:22:12 2020

@author: Enrico Regolin
"""

# This module contains the NN Class used (variable number of linear ReLU layers)
# 2 additional funcitons used for the definition of the loss inside the Net() Class

import numpy as np
import pandas as pd
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
#
# costumized loss function
def weighted_mse_loss(inp, target, weights_torch ):
    #weights_torch
    return torch.sum(weights_torch * (inp - target) ** 2)

#%%
#
# generate weights for error weighing
def generate_weights_dict(condensed_classes):
    
    if condensed_classes:
        weights = [2, 1, 1, 1, 1, 5]
        norm_weights = weights/np.sum(weights)
        weights_dict = dict(zip(range(3,3+len(norm_weights)), norm_weights))
    else:
        weights = [20, 10, 4, 2, 1, 1, 1 ,2, 7, 10]
        norm_weights = weights/np.sum(weights)
        weights_dict = dict(zip(range(1,len(norm_weights)+1), norm_weights))
    
    return weights_dict
   
    
#%%
#
# NN class
class Net(nn.Module):
    def __init__(self,n_inputs=10,n_outputs=1,*argv,**kwargs):
        super().__init__()
    
    # NNs of this class can have a maximum depth of 5 layers
    ## HERE WE PARSE THE ARGUMENTS
    
        self.net_depth = np.maximum(1,len(argv))
        self.n_inputs = n_inputs
        
        if not argv:
            self.n_layer_1 = 1
        else:
            for i,arg in zip(range(1,self.net_depth+1),argv):
                setattr(self, 'n_layer_'+str(i), arg)
        self.n_outputs = n_outputs
    
        self.lr = 0.001 
        self.log_df = pd.DataFrame(columns=['Epoch','Loss Train','Loss Test'])
        self.reshuffle_in_epoch = True
    
        self.update_NN_structure()    
    
    ## let's make the net dynamic by using the "property" decorator
        @property
        def net_depth(self):
            return self._net_depth
        @net_depth.setter
        def net_depth(self,value):
            if value > 5:
                raise AttributeError(f'NN is too deep')
            else:
                self._net_depth = value     
        
        @property
        def n_inputs(self):
            return self._n_inputs
        @n_inputs.setter
        def n_inputs(self,value):
            self._n_inputs = value
            
        @property
        def n_layer_1(self):
            return self._n_layer_1
        @n_layer_1.setter
        def n_layer_1(self,value):
            self._n_layer_1 = value
             
        @property
        def n_layer_2(self):
            return self._n_layer_2
        @n_layer_2.setter
        def n_layer_2(self,value):
            self._n_layer_2 = value     
        
        @property
        def n_layer_3(self):
            return self._n_layer_3
        @n_layer_3.setter
        def n_layer_3(self,value):
            self._n_layer_3 = value     

        @property
        def n_layer_4(self):
            return self._n_layer_4
        @n_layer_4.setter
        def n_layer_4(self,value):
            self._n_layer_4 = value     

        @property
        def n_layer_5(self):
            return self._n_layer_5
        @n_layer_5.setter
        def n_layer_5(self,value):
            self._n_layer_5 = value     

        @property
        def n_outputs(self):
            return self._n_outputs
        @n_outputs.setter
        def n_outputs(self,value):
            self._n_outputs = value     

        @property
        def fc1(self):
            return self._fc1
        @fc1.setter
        def fc1(self,value):
            self._fc1 = value
        
        @property
        def fc2(self):
            return self._fc2
        @fc2.setter
        def fc2(self,value):
            self._fc2 = value
            
        @property
        def fc3(self):
            return self._fc3
        @fc3.setter
        def fc3(self,value):
            self._fc3 = value
        
        @property
        def fc4(self):
            return self._fc4
        @fc4.setter
        def fc4(self,value):
            self._fc4 = value
        
        @property
        def fc5(self):
            return self._fc5
        @fc5.setter
        def fc5(self,value):
            self._fc5 = value
        
        @property
        def fc_output(self):
            return self._fc_output
        @fc_output.setter
        def fc_output(self,value):
            self._fc_output = value


    ## HERE WE DEAL WITH ADDITIONAL (OPTIONAL) ARGUMENTS
        # lr = optimizer learning rate
        default_optimizer = optim.Adam(self.parameters(),self.lr)
        default_loss_function = nn.MSELoss()  #this might be made changable in the future 
        default_weights_dict = {new_list: .1 for new_list in range(1,11)} 
        
        allowed_keys = {'EPOCHS':6,'BATCH_SIZE':200,'device': torch.device("cpu"), \
                        'optimizer' : default_optimizer , 'loss_function' : default_loss_function, \
                        'weights_dict' : default_weights_dict }#, 'VAL_PCT':0.25 , \
                        
        # initialize all allowed keys
        self.__dict__.update((key, allowed_keys[key]) for key in allowed_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        

    ## HERE WE re-DEFINE THE LAYERS when needed        
    def update_NN_structure(self):
        # defining first layer
        self.fc1 = nn.Linear(self.n_inputs, self.n_layer_1)
        # defining layers 2,3,etc.
        for i in range(1,self.net_depth):
            setattr(self, 'fc'+str(i+1), nn.Linear(getattr(self, 'n_layer_'+str(i)),getattr(self, 'n_layer_'+str(i+1) ) ) )
        # define last layer
        last_layer_width = getattr(self, 'n_layer_'+str(self.net_depth)  )
        self.fc_output = nn.Linear(last_layer_width, self.n_outputs)

        
    ## HERE WE DEFINE THE PATH
    # inside the forward function we can create CONDITIONAL PATHS FOR CERTAIN LAYERS!!
    def forward(self,x):
  #      x = F.relu(self.fc1(x))
        iterate_idx = 1
        for attr in self._modules:
            if 'fc'+str(iterate_idx) in attr:
#                set_trace()
                x = F.relu(self._modules[attr](x))
                iterate_idx += 1
                if iterate_idx > self.net_depth:
                    break
        x = self.fc_output(x)
        return x #F.log_softmax(x,dim=1)  # we return the log softmax (sum of probabilities across the classes = 1)
    
    # train the network    
    def train_my_nn(self,train_X, train_y, test_X, test_y):
  
        test_X, test_y = test_X.to(self.device), test_y.to(self.device)
        
        for epoch in tqdm(range(self.EPOCHS)):
           
            if self.reshuffle_in_epoch:
               train_data = torch.cat((train_X, train_y),1)
               indexes = torch.randperm(train_data.shape[0])
               train_data = train_data[indexes] 
               train_X, train_y = torch.split(train_data, [train_X.shape[1],1], 1)
           
            for i in range(0, len(train_X), self.BATCH_SIZE): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
                #print(f"{i}:{i+BATCH_SIZE}")
                batch_X = train_X[i:i+self.BATCH_SIZE] #.view(-1, 1, train_X.shape[1])
                batch_y = train_y[i:i+self.BATCH_SIZE]
                
                
                # apply dictionary "weights_dict" to tensor "target" and obtain tensor of weights
                weights_torch = torch.from_numpy(np.vectorize(self.weights_dict.get)(batch_y.numpy()))
                #set_trace()
                
                weights_torch = weights_torch.to(self.device)
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.zero_grad()

                outputs = self(batch_X)
              #  loss = self.loss_function(outputs, batch_y)
                loss = weighted_mse_loss(outputs, batch_y,weights_torch)
                loss.backward()
                self.optimizer.step()    # Does the update
                
            with torch.no_grad():
                eval_test = self(test_X)
                loss_test = self.loss_function(eval_test, test_y)
            
            new_row = [epoch, loss.item(), loss_test.item()]
            self.log_df.loc[len(self.log_df)] = new_row
            
            #print(f"Epoch: {epoch}. LossTrain: {loss}\n")
            #print(f"Epoch: {epoch}. LossTest : {loss_test}")
    
    # save the network parameters and the training history
    def save_net_params(self,path_log,net_name):
        filename_pt = os.path.join(path_log, net_name + '.pt')
        filename_log = os.path.join(path_log, 'TrainingLog_'+ net_name + '.pkl')
        torch.save(self.state_dict(),filename_pt)
        self.log_df.to_pickle(filename_log)

    # load the network parameters and the training history
    def load_net_params(self,path_log,net_name):    
        filename_pt = os.path.join(path_log, net_name + '.pt')
        filename_log = os.path.join(path_log, 'TrainingLog_'+ net_name + '.pkl')
        self.log_df = pd.read_pickle(filename_log)
        # update net parameters based on state_dict() (which is loaded afterwards)
        self.n_inputs =  (torch.load(filename_pt)['fc1.weight']).shape[1]
        self.net_depth = int(len(torch.load(filename_pt))/2)-1
        for i in range(1,self.net_depth+1):
            setattr(self, 'n_layer_'+str(i), (torch.load(filename_pt)['fc' + str(i) + '.weight']).shape[0])
        self.n_outputs =  (torch.load(filename_pt)['fc_output.weight']).shape[0]
        self.update_NN_structure()        
        self.load_state_dict(torch.load(filename_pt))
        self.to(self.device)  # required step when loading a model on a GPU (even if it was also trained on a GPU)

    # plot the training history
    def plot_training_log(self, init_epoch=0, final_epoch =-1):
        log_norm_df=(self.log_df-self.log_df.min())/(self.log_df.max()-self.log_df.min())
        #fig, ax = plt.subplots()
        fig = plt.figure(figsize=(12, 12))
        #ax = fig.gca()
        ax = log_norm_df[init_epoch:final_epoch][['Loss Train','Loss Test']].plot()
        fig = ax.get_figure()
        return fig
        
        

    # function which runs a prediction on the test set
    def run_test_prediction(self, test_X, test_y, condensed_classes = False, print_acc = True):
        if not torch.is_tensor(test_X):
            test_X = torch.tensor(test_X.astype(np.float32))
            test_y = torch.tensor(test_y.astype(np.float32))
                
        test_X = test_X.to(self.device)
        y_pred = self(test_X).to(self.device)[:,0].view(test_X.shape[0],-1)
            
        device_cpu = torch.device("cpu")
        y_pred.to(device_cpu)
        pred_y = np.round(y_pred.data.cpu().numpy()).astype(float)
    
        y_min = 1
        y_max = 10
        if condensed_classes:
            y_min = 3
            y_max = 8
    
        pred_y = np.minimum(np.maximum(y_min,pred_y), y_max)
    
        #calculate accuracy
        correct_total = [pred_y[i] == test_y.numpy()[i] for i in range(len(pred_y))]
        accuracy = np.sum(correct_total)/len(test_y)
        
        if print_acc:
            print(f'Accuracy = ' + str(round(accuracy*100,2)) +'%')
        
        return pred_y, accuracy