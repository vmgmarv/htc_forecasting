# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:30:38 2023

@author: Marvin Gabriel
"""

from statsmodels.tsa.stattools import kpss
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics import tsaplots
import simdkalman
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


df_train = df.iloc[(df.index >= '2017-01-01') & (df.index <= '2023-01-31'),:]
df_train1 = df_train.iloc[df_train.index <= '2018-10-31',:]
df_train2 = df_train.iloc[df_train.index >= '2019-03-31',:]

df_train = df_train1.append(df_train2)


########## stationarity
for i in df_train.columns:
     print(i)
     x = df_train[i][~pd.isnull(df_train[i])]
     kpss_test(x)
    
#### First difference
#df_train['BVAL_10Y'] = df_train['BVAL_10Y'].diff().diff()
#df_train['BVAL_7Y'] = df_train['BVAL_7Y'].diff().diff()
#df_train['inflation'] = df_train['inflation'].diff()
df_train['m3'] = df_train['m3'].diff()
df_train['m2'] = df_train['m2'].diff()
#df_train['USGG_7Y'] = df_train['USGG_7Y'].diff().diff()
#df_train['USGG_10Y'] = df_train['USGG_10Y'].diff().diff()
df_train['Brent_crude_oil'] = df_train['Brent_crude_oil'].diff()

########## stationarity
for i in df_train.columns:
     print(i)
     x = df_train[i][~pd.isnull(df_train[i])]
     kpss_test(x)

##### get lags
df_lag = df_train.copy()

#df_lag = df_lag.drop(['BSP_rrp','BVAL_7Y','BVAL_10Y','BSP_rrp_est'], axis=1)

############# only select good vars
df_lag = df_lag[['inflation','USGG_7Y','Brent_crude_oil']]

for i in df_lag.columns:
    
    kf = simdkalman.KalmanFilter(
            state_transition = np.array([[1,1],[0,1]]),
            process_noise = np.diag([0.4,0.08]),
            observation_model = np.array([[1,0]]),
            observation_noise = 1.0)
    
    data = df_lag[i].values
    smoothed = kf.smooth(data)
    
    df_lag[i + str('_s')] = smoothed.observations.mean

for i in df_lag.columns[0:]:
    df_lag[i + '_l1'] = df_lag[i].shift(1)
    df_lag[i + '_l2'] = df_lag[i].shift(2)
#    df_lag[i + '_l3'] = df_lag[i].shift(3)
    # df_lag[i + '_l3'] = df_lag[i].shift(4)

##### Forward selection
X = df_lag.dropna()
#X = df_temp.drop(['BVAL_7Y'], axis=1)
y = df_train['BVAL_10Y'].iloc[3:]

y_kalman = kf.smooth(y.values)
y_s = y_kalman.observations.mean

sfs1 = sfs(LinearRegression(),
          k_features=15,
          forward=False,
          floating=False,
          scoring = 'r2',
          cv = 0)


sfs1 = sfs1.fit(X, y_s)
feat_names = list(sfs1.k_feature_names_)
print(feat_names)


############################## new regression, manual
#df_temp = df_lag.dropna()
#df_temp['BVAL_7Y'] = df_temp['BVAL_7Y'].shift(1)
#df_temp = df_temp.dropna()
#
#X = df_temp.drop(['BVAL_7Y'], axis=1)
#y = df_temp['BVAL_7Y']


variables_ = [
#'inflation',
# 'Brent_crude_oil',
# 'inflation_s',
 'USGG_7Y_s',
 'Brent_crude_oil_s',
# 'inflation_l1',
# 'USGG_7Y_l1',
# 'USGG_7Y_l2',
# 'Brent_crude_oil_l1',
# 'inflation_s_l1',
# 'inflation_s_l2',
# 'USGG_7Y_s_l1',
# 'USGG_7Y_s_l2',
# 'Brent_crude_oil_s_l1',
# 'Brent_crude_oil_s_l2'
        ]

X_feat_new = X[variables_]
X_new = sm.add_constant(X_feat_new)

mod = sm.OLS(y_s, X_new)
mod2 = mod.fit()
print(mod2.summary())


# VIF dataframe
vif_data2 = pd.DataFrame()
vif_data2["feature"] = X_feat_new.columns
  
# calculating VIF for each feature
vif_data2["VIF"] = [variance_inflation_factor(X_feat_new.values, i)
                          for i in range(len(X_feat_new.columns))]

print(vif_data2)

###################################### Residual test
## Normality
resid_ = mod2.resid
# stats.probplot(resid_,dist='norm', plot=pylab)

## shapiro
stat, p = shapiro(resid_)
print('shapiro-wilk')
print('stat = %.3f, p = %.3f \n' % (stat,p))
if p > 0.05:
    print('Probably Gaussian')

else:
    print('Probably not Gaussian')

## Jarque Berra
print('JB')
stat3, p3 = jarque_bera(resid_)
print('stat = %.3f, p = %.3f \n' % (stat3,p3))
if p3 > 0.05:
    print('Probably Gaussian')

else:
    print('Probably not Gaussian')
    
    
    ## Durbin watson
print('Durbin watson ', durbin_watson(resid_))

## Breusch Pagan
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']

BP = sms.het_breuschpagan(resid_, mod2.model.exog)

print(lzip(names, BP))

'''
CO for serial cor
'''
################# CO for serial cor
model_co = cochrane_orcutt.OLSAR1(mod2)
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
    
## Jarque Berra
print('JB')
stat3, p3 = jarque_bera(resid_)
print('stat = %.3f, p = %.3f \n' % (stat3,p3))
if p3 > 0.05:
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


########################### ACF residual
resid_acf = sm.tsa.acf(resid_2,nlags=10)
fig4 = tsaplots.plot_acf(resid_acf, lags=10)
plt.show()


############################# In-sample_validation
###
#df_in = X_feat_new.copy()

df_in = df_train[['BVAL_10Y','USGG_7Y', 'Brent_crude_oil']]

df_in['Forecast'] = 3.18559401 + 0.94541024 * df_in.USGG_7Y - 0.02556627*df_in.Brent_crude_oil
df_in['BVAL_7Y'] = y
#
df_in['APE'] = abs((df_in.BVAL_10Y - df_in.Forecast)/ df_in.BVAL_10Y)
print('MAPE insample ', df_in.APE.mean())
###
###fig, axs = plt.subplots(1,figsize=(17,12))
###plt.plot(df_in.Forecast,marker = 'o', label = 'Predictions in-sample')
###plt.plot(df_in.BVAL_7Y,marker = 'o', label = 'Actual in-sample')
###plt.legend(fontsize = 15)
###plt.tick_params(axis='x',labelsize=13)
###plt.tick_params(axis='y',labelsize=13)
#### plt.ylim(0,0.1)
###axs.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))
###
############################################### out-sample_validation
df_test = df.iloc[df.index > '2022-12-28']
for i in df_test.columns[1:]:
    df_test[i + '_l1'] = df_test[i].shift(1)
    df_test[i + '_l2'] = df_test[i].shift(2)

#variables_.append('BVAL_10Y')
df_test = df_test[['BVAL_10Y','USGG_7Y', 'Brent_crude_oil']]

#vars_out = ['BVAL_7Y',  'm2',
#            'USGG_7Y',
#            'FED_FFR_l1'] 
###
df_out = df_test.copy()
df_out = df_out.dropna()
df_out['Forecast'] = 3.18559401 + 0.94541024 * df_out.USGG_7Y - 0.02556627*df_out.Brent_crude_oil

df_out = df_out.iloc[df_out.index > '2023-01-31',:]

#
df_out['APE'] = abs((df_out.BVAL_10Y - df_out.Forecast)/ df_out.BVAL_10Y)
print('MAPE outsample ', df_out.APE.mean())
