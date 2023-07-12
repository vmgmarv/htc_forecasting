# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:52:51 2023

@author: Marvin Gabriel
"""


import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, kstest, jarque_bera
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
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




#exog_features = ['inflation', 'm3', 'USGG_7Y',
#       'FED_FFR', 'Brent_crude_oil']

df_train = df.iloc[(df.index >= '2016-01-01') & (df.index <= '2023-01-31'),:]
#df_train1 = df_train.iloc[df_train.index <= '2020-06-30',:]
#df_train2 = df_train.iloc[df_train.index >= '2021-03-30',:]

#df_train = df_train1.append(df_train2)


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
#df_train['Brent_crude_oil'] = df_train['Brent_crude_oil'].diff()

################################################### ARIMAX

exog_features = ['inflation','m3','USGG_7Y']
comp_features = ['BVAL_7Y','inflation','m3','USGG_7Y']

df_train = df_train[comp_features].dropna()
#model = pm.auto_arima(df_train.BVAL_7Y, 
#                      exogenous=df_train[exog_features], 
##                      test = 'adf',
##                      start_p = 2, start_q = 2,
##                      max_p = 10, max_q = 10,
##                      seasonal = False,
##                      D=1,
##                      trace=True, 
##                      suppress_warnings=True,
##                      stepwise=True
#                      order = (1,1,1)
#                        )

model3=sm.tsa.ARIMA(endog=df_train.BVAL_7Y,
                    exog=df_train[exog_features],order=[2,1,2])


#model.fit(df_train.BVAL_7Y, exogeneous=df_train[exog_features])

res = model3.fit()
#
#
df_test = df.iloc[df.index > '2022-12-30']
df_test['m3'] = df_test['m3'].diff()
df_test['m2'] = df_test['m2'].diff()

df_test = df_test[exog_features]
df_test = df_test.dropna()

df_train = df_train[exog_features]
df_res = df_train.append(df_test)

#forecast = res.forecast(4,exog = df_test
##                            start = df_test.index.values[0],
##                            end = df_test.index.values[-1],
##                            exog= df_test[exog_features]
#                        )




#forecast = model.predict(n_periods = len(df_train),
#                         exog=df_train[exog_features])
#df_test["Forecast_ARIMAX"] = forecast.values

#df_train[["BVAL_7Y", "Forecast_ARIMAX"]].plot(figsize=(14, 7))


#res.plot_diagnostics(figsize=(15,12))
########################## ACF residual
#resid_acf = sm.tsa.acf(resid_,nlags=10)
#fig4 = tsaplots.plot_acf(resid_acf, lags=10)
#plt.show()



