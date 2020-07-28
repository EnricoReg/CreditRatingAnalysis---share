# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:04:00 2020

@author: Enrico Regolin
"""

import numpy as np
import pandas as pd
import pickle
import os

import torch
import matplotlib.pyplot as plt

# This module contains and classes needed to load data and display results


#%%

def print_accuracies(y_test, y_pred):
    
    pred_error = np.reshape(y_test,[-1,1]) - np.reshape(y_pred,[-1,1])
    
    # Accuracy 0
    class_error_0 = round(100* (pred_error==0).sum() /len(pred_error),2)
    
    # Accuracy 1
    class_error_1 = round(100* (abs(pred_error)<=1).sum() /len(pred_error),2)
    
    # print Accuracies
    print('Accuracy_0: Percentage of instances where exact class is detected')
    print('Accuracy_1: Percentage of instances where exact class or neighbouring class is detected')
    print(' ')
    print(f'Accuracy_0 = {class_error_0}%')
    print(f'Accuracy_1 = {class_error_1}%')




#%%
#
# print confusion matrix
def plot_confusion_matrix(test_y, pred_y, condensed_classes):

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    font_size = 20

    if condensed_classes:
        target_names = ['AAA/AA/A', 'BBB', 'BB', 'B', 'CCC', 'CC/C/D'][::-1]
    else:
        target_names = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C','D'][::-1]

    if torch.is_tensor(test_y):
        test_y = test_y.numpy()
        
    bool_mask_test = ~np.isnan(test_y)
    bool_mask_pred = ~np.isnan(pred_y)
    test_y__ = test_y[bool_mask_test]
    pred_y__ = pred_y[bool_mask_pred]

    pred_y__ = pred_y__.reshape((len(pred_y__),1))
    test_y__ = test_y__.reshape((len(pred_y__),1))

    cm = confusion_matrix(test_y__, pred_y__)
    # Normalise
    cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)


#%%
#
# class used to load the proper data to be used to train the NN
class DataHandler():
    
    def __init__(self):
#        self.year = None

# these attributes have conditions one vs. the other, threfore we define getter/setter properties for them, 
#to include conditional evaluations 
        self.complete_only = False
        self.incomplete_only = False
        self.full_db = True
        self.fill_nans_as_zeros = False
        self.remove_incomplete_y = True

#        self.NACE_parent = False        #outputs NACE parent code among features
        self.ratios_db = True          #if True: uses ratios DB instead
        self.imputed_db = False  # only evaluated in case ratios_db==True
        
        self.condensed_classes = False  #if True: uses unique rating classes for AAA/AA/A and CC/C/D
        self.SME_only = False

        self.normalize_features = True
        self.NACE_classification = True
        
        self.path_db = os.path.abspath(os.getcwd())+'\\db_generation\\db'


    @property
    def full_db(self):
        return self._full_db
        
    @property
    def complete_only(self):
        return self._complete_only
    
    @property
    def incomplete_only(self):
        return self._incomplete_only
     
    @full_db.setter
    def full_db(self, value):
       # set_trace()
        if self.complete_only or self.incomplete_only:
            self._full_db = False
        else:
            self._full_db = value

    @complete_only.setter
    def complete_only(self, value):
        self._complete_only = value
        # in case the attribute full_db is already defined, we have to re-evaluate it 
        #in light of the assignment of incomplete only
        if hasattr(self,'full_db'):
            self.full_db = self.full_db
        if hasattr(self,'incomplete_only'):
            if self.incomplete_only and self.complete_only:
                raise ValueError("complete_only and incomplete_only flags both =1")
                

    @incomplete_only.setter
    def incomplete_only(self, value):
        self._incomplete_only = value
        # in case the attribute full_db is already defined, we have to re-evaluate it 
        #in light of the assignment of incomplete only
        if hasattr(self,'full_db'):
            self.full_db = self.full_db
        if hasattr(self,'complete_only'):
            if self.incomplete_only and self.complete_only:
                raise ValueError("complete_only and incomplete_only flags both =1")
    
    def load_as_pandas(self):
        # load features
        if self.ratios_db:
            if self.imputed_db:
                df_X = pd.read_pickle(os.path.join(self.path_db, 'features_ratio_imputed.pkl'))
            else:    
                df_X = pd.read_pickle(os.path.join(self.path_db, 'features_ratio.pkl'))        
        else:
            df_X = pd.read_pickle(os.path.join(self.path_db, 'features_original.pkl'))
            
        if not self.NACE_classification:
            df_X = df_X.loc[:,~df_X.columns.str.contains('NACE')]

            

        # load labels    
        if self.condensed_classes:
            df_y = pd.read_pickle(os.path.join(self.path_db, 'rating_reduced.pkl'))
        else:
            df_y = pd.read_pickle(os.path.join(self.path_db, 'rating_all_classes.pkl'))
        
        # filter in case of SME only
        if self.SME_only:
            index_SME = df_X['Total assets']<43000
            df_X = df_X[index_SME].copy()
            df_y = df_y[index_SME].copy()
            df_X.reset_index(drop = True, inplace = True) 
            df_y.reset_index(drop = True,inplace = True) 

        # define "full" and "complete-only" databases
        df_X_complete = df_X.dropna(how = 'any').copy()
        df_y_complete = df_y.iloc[df_X_complete.index]
        df_y_complete = df_y_complete.dropna(how = 'any').copy()
        df_X_complete = df_X.iloc[df_y_complete.index]
        df_X_incomplete = df_X.drop(df_X_complete.index)
        df_y_incomplete = df_y.drop(df_y_complete.index)
        
        if self._full_db:
            if self.remove_incomplete_y:
                df_y = df_y.dropna(how = 'any').copy()
                df_X = df_X.iloc[df_y.index]
            df_X_out = df_X
            df_y_out = df_y
            if self.fill_nans_as_zeros:
                df_X_out = df_X_out.fillna(0)
                df_y_out = df_y_out.fillna(0)
            
        elif self._complete_only:
            # select matrices
            df_X_out = df_X_complete
            df_y_out = df_y_complete
        
        elif self._incomplete_only:
            df_X_out = df_X_incomplete
            df_y_out = df_y_incomplete
            if self.fill_nans_as_zeros:
                df_X_out.fillna(0)
                df_y_out.fillna(0)
            
        if self.normalize_features:
            df_X_out = (df_X_out - df_X_out.min()) * 1.0 / (df_X_out.max() - df_X_out.min() )
        
        return df_X_out, df_y_out
        
        
    def load_as_numpy(self):

        df_X,df_y = self.load_as_pandas()
        X = df_X.values
        y = df_y.values
        
        return X,y


#%%
#
# class used to convert a dataset into the proper torch format
class TorchDatasetsGenerator():
    def __init__(self,VAL_PCT = 0.2, shuffle_on = True):
        self.VAL_PCT = VAL_PCT
        self.shuffle_on = shuffle_on 
        
    def generate_torch_dataset(self,X,y):
        if self.shuffle_on:
            while True:
                training_data = [X,y]
                np.random.shuffle(training_data)
                X_shf,y_shf = training_data
                if X_shf.shape == X.shape:
                    X = X_shf
                    y = y_shf
                    break
                    
        X_torch = torch.tensor(X).view(-1,X.shape[1])
        y_torch = torch.tensor(y)

        val_size = int(X.shape[0]*self.VAL_PCT)
        
        train_X = X_torch[:-val_size].float()
        train_y = y_torch[:-val_size].float()
        test_X = X_torch[-val_size:].float()
        test_y = y_torch[-val_size:].float()
        
        return train_X, train_y, test_X, test_y