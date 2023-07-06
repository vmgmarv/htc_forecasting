# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 22:45:36 2022

@author: 213606
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
import feature_selection
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import cochrane_orcutt
import matplotlib.dates as md
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
# from sklearn.linear_model import LinearRegression
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

# df = df.drop('BSP_rrp', axis=1)

# df = df[df.index >= '2018-01-01']


df_train = df.iloc[0:81]
df_test = df.iloc[81:88]

########### stationarity
# for i in df_train.columns:
#     print(i)
#     x = df_train[i][~pd.isnull(df_train[i])]
#     kpss_test(x)
    
### First difference
df_train['m3'] = df_train['m3'].diff()
df_train['m2'] = df_train['m2'].diff()

########## stationarity
# for i in df.columns:
#     print(i)
#     x = df[i][~pd.isnull(df[i])]
#     kpss_test(x)

##### get lags
df_lag = df_train.copy()

df_lag = df_lag.drop(['BSP_rrp_est','BVAL_7Y','BSP_rrp'], axis=1)

for i in df_lag.columns[1:]:
    df_lag[i + '_l1'] = df_lag[i].shift(1)
    df_lag[i + '_l2'] = df_lag[i].shift(2)
    # df_lag[i + '_l3'] = df_lag[i].shift(3)
    # df_lag[i + '_l3'] = df_lag[i].shift(4)

##### Forward selection
df_temp = df_lag.dropna()
X = df_temp.drop(['BVAL_7Y'], axis=1)
y = df_train['BVAL_7Y'].iloc[3:]

# X = df_temp

feat_names = list(forward_selection(X, y))
# print(forward_selection(X, y))


# ############################## new regression, manual

X_feat = df_temp[['inflation_l2', 'USGG_10Y_l1']]
X_new = sm.add_constant(X_feat)

mod = sm.OLS(y, X_new).fit()
print(mod.summary())


# VIF dataframe
vif_data2 = pd.DataFrame()
vif_data2["feature"] = X_feat.columns
  
# calculating VIF for each feature
vif_data2["VIF"] = [variance_inflation_factor(X_feat.values, i)
                          for i in range(len(X_feat.columns))]

print(vif_data2)


resid_1 = mod.resid


## shapiro
stat, p = shapiro(resid_1)
print('shapiro-wilk')
print('stat = %.3f, p = %.3f \n' % (stat,p))
if p > 0.05:
    print('Probably Gaussian')

else:
    print('Probably not Gaussian')


## Jarque Berra
print('JB')
stat3, p3 = jarque_bera(resid_1)
print('stat = %.3f, p = %.3f \n' % (stat3,p3))
if p3 > 0.05:
    print('Probably Gaussian')
    
    
## Durbin watson
print('Durbin watson ', durbin_watson(resid_1))

## Breusch Pagan
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']

BP = sms.het_breuschpagan(resid_1, mod.model.exog)

print(lzip(names, BP))


'''
CO for serial cor
'''
################## CO for serial cor
model_co = cochrane_orcutt.OLSAR1(mod)
print(model_co.summary())


resid_2 = model_co.resid

## shapiro
stat, p = shapiro(resid_2)
print('shapiro-wilk')
print('stat = %.3f, p = %.3f \n' % (stat,p))
if p > 0.05:
    print('Probably Gaussian')

else:
    print('Probably not Gaussian')
    
## Durbin watson
print('Durbin watson ', durbin_watson(resid_2))

## Breusch Pagan
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']

BP = sms.het_breuschpagan(resid_2, model_co.model.exog)

print(lzip(names, BP))


# # ########################## ACF residual
# # resid_acf = sm.tsa.acf(resid_,nlags=10)
# # fig4 = tsaplots.plot_acf(resid_acf, lags=10)
# # plt.show()


# # ########################## In-sample_validation

# df_in = X_feat.copy()
# df_in['Forecast'] = 2.52819948 + 0.73626461 * df_in.usgg_l1 + 0.23927266 * df_in.inflation_l2
# df_in['BVAL_7Y'] = y

# df_in['APE'] = abs((df_in.BVAL_7Y - df_in.Forecast)/ df_in.BVAL_7Y)
# print('MAPE insample ', df_in.APE.mean())

# # fig, axs = plt.subplots(1,figsize=(17,12))
# # plt.plot(df_in.Forecast,marker = 'o', label = 'Predictions in-sample')
# # plt.plot(df_in.BVAL_7Y,marker = 'o', label = 'Actual in-sample')
# # plt.legend(fontsize = 15)
# # plt.tick_params(axis='x',labelsize=13)
# # plt.tick_params(axis='y',labelsize=13)
# # # plt.ylim(0,0.1)
# # axs.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))

# ########################## out-sample_validation
# df_test = df_test[['BVAL_7Y','usgg','inflation']]
# for i in df_test.columns[1:]:
#     df_test[i + '_l1'] = df_test[i].shift(1)
#     df_test[i + '_l2'] = df_test[i].shift(2)
    
# df_out = df_test[['BVAL_7Y','usgg_l1', 'inflation_l2']]
# df_out = df_out.dropna()
# df_out['Forecast'] = 2.52819948 + 0.73626461 * df_out.usgg_l1 + 0.23927266 * df_out.inflation_l2
# df_out['APE'] = abs((df_out.BVAL_7Y - df_out.Forecast)/ df_out.BVAL_7Y)
# print('MAPE outsample ', df_out.APE.mean())

# # fig2, axs = plt.subplots(1,figsize=(17,12))
# # plt.plot(df_out.Forecast,marker = 'o', label = 'Predictions out-sample')
# # plt.plot(df_out.BVAL_7Y,marker = 'o', label = 'Actual out-sample')
# # plt.legend(fontsize = 15)
# # plt.tick_params(axis='x',labelsize=13)
# # plt.tick_params(axis='y',labelsize=13)
# # # plt.ylim(0,0.1)
# # axs.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))


# ################################### combine in and out

# df_result = df_in.append(df_out)

# fig, axs = plt.subplots(1,figsize=(17,12))
# plt.plot(df_result.Forecast,marker = 'o', label = 'Predicted')
# plt.plot(df_result.BVAL_7Y,marker = 'o', label = 'Actual')
# plt.tick_params(axis='x',labelsize=13)
# plt.tick_params(axis='y',labelsize=13)
# axs.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))
# plt.axvspan('2022-04-30', '2022-06-30', alpha = 0.5, 
#             color='gray',label='Out-of_sample')
# plt.legend(fontsize = 15)
# plt.ylabel('BVAL 7Y', fontsize=13)
