# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 08:49:00 2023

@author: Marvin Gabriel
"""

from statsmodels.tsa.stattools import kpss
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics import tsaplots
# import scipy.stats as stats
from scipy.stats import shapiro, kstest, jarque_bera
from statsmodels.stats.stattools import durbin_watson
from statsmodels.compat import lzip
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import cochrane_orcutt
import matplotlib.dates as md
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#import shap
# from sklearn.metrics import mean_absolute_error as mae
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")


def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d

def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    # print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    # print(f'num lags: {n_lags}')
    # print('Critial Values:')
    # for key, value in critical_values.items():
    #     print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    print('###########################')
    
def forward_selection(data, target, significance_level=0.10):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        
        min_p_value = new_pval.min()
        if(min_p_value < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
        
    return best_features
    
    


df = pd.read_csv('dataset_new.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

year_start = df.year.values[0].astype(int)
year_end = df.year.values[-1].astype(int)
month_start = df.month.values[0].astype(int)
month_end = df.month.values[-1].astype(int)

try:    
    dates = month_iterator('{}-{}-01'.format(year_start,month_start), '{}-{}-01'.format(year_end,month_end+1))       
except:
    dates = month_iterator('{}-{}-01'.format(year_start,month_start), '{}-{}-01'.format(year_end+1,1))
df['Dates'] = dates
df = df.drop(['month','year'],axis=1)


df['Dates']=pd.to_datetime(df['Dates'])

df = df.set_index('Dates')

#################################################### log transform
#df['BVAL_7Y'] = np.log(df['BVAL_7Y'])

# df = df.drop('BSP_rrp', axis=1)

#df = df[df.index >= '2017-01-01']

df['m3'] = df['m3'].diff()
df['m2'] = df['m2'].diff()

df['inflation_l1'] = df['inflation'].shift(1)


df_train = df.iloc[(df.index >= '2016-01-01') & (df.index <= '2023-01-31'),:]
df_train1 = df_train.iloc[df_train.index <= '2018-10-31',:]
df_train2 = df_train.iloc[df_train.index >= '2019-03-31',:]

df_train = df_train1.append(df_train2)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df_train.BVAL_10Y, '-', label = 'BVAL_7Y',color = 'orange')
ax.plot(df_train.inflation_l1, '-', label = 'inflation_l1',color='red')
ax.plot(df.USGG_7Y, '-', label = 'USGG_7Y',color='violet')
#ax.plot(df.USGG_10Y, '-', label = 'USGG_10Y',color='yellow')
#ax.plot(df.FED_FFR, '-', label = 'FED_FFR',color='pink')

ax2 = ax.twinx()
#ax2.plot(df.m3, '-', label = 'm3',color = 'green')
ax2.plot(df_train.Brent_crude_oil, '-', label = 'Brent_crude_oil',color = 'black')
ax.legend(loc=0)
ax2.legend(loc=1)
ax.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))


