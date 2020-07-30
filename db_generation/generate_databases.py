# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:45:41 2020

@author: Enrico Regolin
"""


import numpy as np
import pandas as pd
import os

from EM_imputation import *

#%%
#
path = os.path.abspath(os.getcwd())

if os.name == 'nt':
	path_db = path + '\\db'
else: 
	path_db = path + '/db'

save_database	      = True
save_database_no_year = True
save_database_ratios = True

run_imputation = True
save_imputed_db = True 


#%%
##########################################################################################
##########################################################################################
##########################################################################################
# GENERATE STANDARD FULL DATABASE


#%% 
# import NACE codes db

df_NACE = pd.read_csv('codes_NACE.csv', delimiter=";")
#df_NACE_parent = pd.read_csv('Parent_NACE.csv', delimiter=";")
df_NACE.Code = df_NACE.Code.astype(str)


#%%
# import MF database and adjust datatypes

df = pd.read_csv('book_2.csv', delimiter=";")
df = df.replace('n.a.',np.nan)
df = df.replace(' ','')

df__= df.loc[:,'Fixed assets 2018':'Total shareh. funds & liab. 2016']
df__ = df__.fillna(-1)
df__ = df__.astype(np.float)
df__ = df__.round().astype(np.int64)
df__ = df__.replace(-1, np.nan)

df.loc[:,'Fixed assets 2018':'Total shareh. funds & liab. 2016'] = df__

#%%
#

column_data = df['NACE Rev. 2 primary code'].apply(lambda x: np.floor(x/100).astype(np.int64).astype(str))
df.insert(1, "NACE root", column_data)

column_data_ = df['NACE root'].map(df_NACE.set_index('Code')['Parent'])
df.insert(1, 'NACE Parent', column_data_)

#%%
# use dictionary for rating classes conversion
keys   = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
values = 1+np.arange(10)
values = values[::-1]
values = values.astype(np.int64)
dictionary = dict(zip(keys, values))

df['MORE evaluation - Score 2016'] = df['MORE evaluation - Score 2016'].map(dictionary)
df['MORE evaluation - Score 2017'] = df['MORE evaluation - Score 2017'].map(dictionary)
df['MORE evaluation - Score 2018'] = df['MORE evaluation - Score 2018'].map(dictionary)

#%%
# drop obsolete columns

df = df.drop(['NACE Rev. 2 primary code','NACE Rev. 2, primary code description'], 1)
# now the full database is complete

#%%
if save_database : 
    filename = os.path.join(path_db, 'full_database.pkl')
    df.to_pickle(filename)
    # this one will not be used, as it contains the company names, and NACE codes are not one-hot encoded

#%%


##########################################################################################
##########################################################################################
##########################################################################################
# GENERATE EXTENDED DATABASE - NO YEAR DEPENDENCIES --> COMPANY NAMES ARE DROPPED, EVERY YEAR IS AN INDEPENDENT ENTRY

#%%
#

# generate global database removing company names and year dependencies
# masks to filter dataframe based on year
map_2016 = df.columns[  ( df.columns.to_series().str.contains('2016') | df.columns.to_series().str.contains('Company name')| df.columns.to_series().str.contains('NACE Parent')).values]
map_2017 = df.columns[  ( df.columns.to_series().str.contains('2017') | df.columns.to_series().str.contains('Company name')| df.columns.to_series().str.contains('NACE Parent')).values]
map_2018 = df.columns[  ( df.columns.to_series().str.contains('2018') | df.columns.to_series().str.contains('Company name')| df.columns.to_series().str.contains('NACE Parent')).values]

a_2016 = df.iloc[:][map_2016]
a_2017 = df.iloc[:][map_2017]
a_2018 = df.iloc[:][map_2018]

a_2016__ = pd.get_dummies(a_2016, columns=['NACE Parent'])
a_2017__ = pd.get_dummies(a_2017, columns=['NACE Parent'])
a_2018__ = pd.get_dummies(a_2018, columns=['NACE Parent'])

a_2016__.columns = a_2016__.columns.str.rstrip(' 2016')
a_2017__.columns = a_2017__.columns.str.rstrip(' 2017')
a_2018__.columns = a_2018__.columns.str.rstrip(' 2018')

frames = [a_2016__, a_2017__, a_2018__]
df_no_year = pd.concat(frames, ignore_index = True)

df_no_year.drop('MORE evaluation - Score', axis=1, inplace=True)
df_no_year.drop('Company name', axis=1, inplace=True)

#%%
if save_database_no_year : 
    filename = os.path.join(path_db, 'features_original.pkl')
    df_no_year.to_pickle(filename)

#%%


##########################################################################################
##########################################################################################
##########################################################################################
# GENERATE RATIOS DATABASE - (NO YEAR DEPENDENCIES)

#%%

df["NetIncomeProxy 2016"] = np.nan
df["NetIncomeProxy 2017"] = df["Total shareh. funds & liab. 2017"]-df["Total shareh. funds & liab. 2016"]
df["NetIncomeProxy 2018"] = df["Total shareh. funds & liab. 2018"]-df["Total shareh. funds & liab. 2017"]

df['Debt 2016'] = df['Long term debt 2016']+df['Loans 2016']
df['Debt 2017'] = df['Long term debt 2017']+df['Loans 2017']
df['Debt 2018'] = df['Long term debt 2018']+df['Loans 2018']

df['Current Ratio 2016'] = df['Current assets 2016'] / df['Current liabilities 2016']
df['Current Ratio 2017'] = df['Current assets 2017'] / df['Current liabilities 2017']
df['Current Ratio 2018'] = df['Current assets 2018'] / df['Current liabilities 2018']

df['Quick Ratio 2016'] = (df['Cash & cash equivalent 2016']+ df['Debtors 2016']) / df['Current liabilities 2016']
df['Quick Ratio 2017'] = (df['Cash & cash equivalent 2017']+ df['Debtors 2017']) / df['Current liabilities 2017']
df['Quick Ratio 2018'] = (df['Cash & cash equivalent 2018']+ df['Debtors 2018']) / df['Current liabilities 2018']

df['Cash Ratio 2016'] = df['Cash & cash equivalent 2016'] / df['Current liabilities 2016']
df['Cash Ratio 2017'] = df['Cash & cash equivalent 2017'] / df['Current liabilities 2017']
df['Cash Ratio 2018'] = df['Cash & cash equivalent 2018'] / df['Current liabilities 2018']

df['ROA 2016'] = df['NetIncomeProxy 2016'] / (df['Fixed assets 2016'] + df['Current assets 2016'])
df['ROA 2017'] = df['NetIncomeProxy 2017'] / (df['Fixed assets 2017'] + df['Current assets 2017'])
df['ROA 2018'] = df['NetIncomeProxy 2018'] / (df['Fixed assets 2018'] + df['Current assets 2018'])

num2016 = df['NetIncomeProxy 2016'].shape
den2016 = np.maximum(0,df['Shareholders funds 2016'])
num2017 = df['NetIncomeProxy 2017'].shape
den2017 = np.maximum(0,df['Shareholders funds 2017'])
num2018 = df['NetIncomeProxy 2018'].shape
den2018 = np.maximum(0,df['Shareholders funds 2018'])

df['ROE 2016'] = num2016/den2016
df['ROE 2017'] = num2017/den2017
df['ROE 2018'] = num2018/den2018

df['ROIC 2016'] = df['NetIncomeProxy 2016'] / (df['Capital 2016'])
df['ROIC 2017'] = df['NetIncomeProxy 2017'] / (df['Capital 2017'])
df['ROIC 2018'] = df['NetIncomeProxy 2018'] / (df['Capital 2018'])

df['D/E 2016'] = df['Debt 2016'] / np.maximum(0.01,df['Shareholders funds 2016'])
df['D/E 2017'] = df['Debt 2017'] / np.maximum(0.01,df['Shareholders funds 2017'])
df['D/E 2018'] = df['Debt 2018'] / np.maximum(0.01,df['Shareholders funds 2018'])

df['Tang/Total Assets 2016'] = df['Fixed assets 2016'] / df['Total assets 2016']
df['Tang/Total Assets 2017'] = df['Fixed assets 2017'] / df['Total assets 2017']
df['Tang/Total Assets 2018'] = df['Fixed assets 2018'] / df['Total assets 2018']

df['CCC proxi 2016'] = ( df['Stock 2016'] + df['Debtors 2016']- df['Creditors 2016'] ) / df['Total assets 2016']
df['CCC proxi 2017'] = ( df['Stock 2017'] + df['Debtors 2017']- df['Creditors 2017'] ) / df['Total assets 2017']
df['CCC proxi 2018'] = ( df['Stock 2018'] + df['Debtors 2018']- df['Creditors 2018'] ) / df['Total assets 2018']

df_ratios = df[[     'Company name', 'NACE Parent', 'Total assets 2016','Total assets 2017','Total assets 2018','Current Ratio 2016','Current Ratio 2017','Current Ratio 2018','Quick Ratio 2016','Quick Ratio 2017','Quick Ratio 2018','Cash Ratio 2016','Cash Ratio 2017','Cash Ratio 2018','ROA 2016','ROA 2017','ROA 2018','ROE 2016','ROE 2017','ROE 2018','D/E 2016','D/E 2017','D/E 2018','ROIC 2016','ROIC 2017','ROIC 2018','Tang/Total Assets 2016','Tang/Total Assets 2017','Tang/Total Assets 2018','CCC proxi 2016','CCC proxi 2017','CCC proxi 2018','MORE evaluation - Score 2016','MORE evaluation - Score 2017','MORE evaluation - Score 2018']]

df_ratios = df_ratios.replace([np.inf,-np.inf],np.nan)

# masks to filter dataframe based on year
## | df_ratios.columns.to_series().str.contains('NACE root') 

map_2016 = df_ratios.columns[  ( df_ratios.columns.to_series().str.contains('2016') | df_ratios.columns.to_series().str.contains('Company name')| df_ratios.columns.to_series().str.contains('NACE Parent')).values]
map_2017 = df_ratios.columns[  ( df_ratios.columns.to_series().str.contains('2017') | df_ratios.columns.to_series().str.contains('Company name')| df_ratios.columns.to_series().str.contains('NACE Parent')).values]
map_2018 = df_ratios.columns[  ( df_ratios.columns.to_series().str.contains('2018') | df_ratios.columns.to_series().str.contains('Company name')| df_ratios.columns.to_series().str.contains('NACE Parent')).values]

a_2016 = df_ratios.iloc[:][map_2016]
a_2017 = df_ratios.iloc[:][map_2017]
a_2018 = df_ratios.iloc[:][map_2018]

a_2016__ = pd.get_dummies(a_2016, columns=['NACE Parent'])
a_2017__ = pd.get_dummies(a_2017, columns=['NACE Parent'])
a_2018__ = pd.get_dummies(a_2018, columns=['NACE Parent'])

a_2016__.columns = a_2016__.columns.str.rstrip(' 2016')
a_2017__.columns = a_2017__.columns.str.rstrip(' 2017')
a_2018__.columns = a_2018__.columns.str.rstrip(' 2018')

frames = [a_2016__, a_2017__, a_2018__]
df_ratio = pd.concat(frames, ignore_index = True)

df_X = df_ratio.copy()
df =  df_X.copy()
df['Rating']=df_ratio['MORE evaluation - Score'].values
df.head()

del df_X['MORE evaluation - Score']
del df_X['Company name']

df_y = pd.DataFrame(data=df_ratio['MORE evaluation - Score'].values, index=df_X.index, columns=['Rating'])

# group credit ratings AAA/AA/A, CC/C/D
df_y_red = df_y.replace(1,3)
df_y_red = df_y_red.replace(2,3)
df_y_red = df_y_red.replace(9,8)
df_y_red = df_y_red.replace(10,8)

df_y_red['Rating'].value_counts()



#%%
if save_database_ratios: 
    filename_ratios = os.path.join(path_db, 'features_ratio.pkl')
    filename_labels_reduced = os.path.join(path_db, 'rating_reduced.pkl')
    filename_labels = os.path.join(path_db, 'rating_all_classes.pkl')
    df_X.to_pickle(filename_ratios)
    df_y.to_pickle(filename_labels)
    df_y_red.to_pickle(filename_labels_reduced)

#%%

##########################################################################################
##########################################################################################
##########################################################################################
# GENERATE IMPUTED DATABASE 


#%%
# EM imputation

if run_imputation:

    X_incomplete = df_X[['Total assets', 'Current Ratio', 'Quick Ratio', 'Cash Ratio', 'ROA',
           'ROE', 'D/E', 'ROIC', 'Tang/Total Assets', 'CCC proxi']].values
    
    
    
    imp_result = impute_em(X_incomplete, max_iter = 120, eps = 0.1)
    X_imp = imp_result['X_imputed']
    
    
    
    df_X_imp_nace = df_X
    df_X_imp_nace.loc[:,'Total assets':'CCC proxi'] = X_imp
    if save_imputed_db:
        filename = os.path.join(path_db, 'features_ratio_imputed.pkl')
        df_X_imp_nace.to_pickle(filename)
