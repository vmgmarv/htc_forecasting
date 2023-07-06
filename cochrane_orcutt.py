# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:58:23 2022

@author: Marvin Gabriel
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm



# cochrane-orcutt / prais-winsten with given AR(1) rho, 
# derived from ols model, default to cochrane-orcutt 
def ols_ar1(model,rho,drop1=True):
    x = model.model.exog
    y = model.model.endog
    ystar = y[1:]-rho*y[:-1]
    xstar = x[1:,]-rho*x[:-1,]
    if drop1 == False:
        ystar = np.append(np.sqrt(1-rho**2)*y[0],ystar)
        xstar = np.append([np.sqrt(1-rho**2)*x[0,]],xstar,axis=0)
    model_ar1 = sm.OLS(ystar,xstar).fit()
    return(model_ar1)

# cochrane-orcutt / prais-winsten iterative procedure
# default to cochrane-orcutt (drop1=True)
def OLSAR1(model,drop1=True):
    x = model.model.exog
    y = model.model.endog
    e = y - (x @ model.params)
    e1 = e[:-1]; e0 = e[1:]
    rho0 = np.dot(e1,e[1:])/np.dot(e1,e1)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1 = ols_ar1(model,rho0,drop1)
        e = y - (x @ model1.params)
        e1 = e[:-1]; e0 = e[1:]
        rho1 = np.dot(e1,e[1:])/np.dot(e1,e1)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        print('Rho = ', rho0)
    # pint final iteration
    # print(sm.OLS(e0,e1).fit().summary())
    model1 = ols_ar1(model,rho0,drop1)
    return(model1)


if __name__ == "__main__":
    
    data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', nrows=39)
    
    X = np.log(data.X)
    L = np.log(data.L1)
    K = np.log(data.K1)
    Z = pd.concat([L,K],axis=1)
    Z = sm.add_constant(Z)
    model = sm.OLS(X,Z).fit()
    
    # # AR(1) based on cochrane-orcutt iterative procedure   
    # ar1_co = OLSAR1(model_ols)
    # # ar1_co = OLSAR1(model_ols,drop1=True)
    # print(ar1_co.summary())
    
    x = model.model.exog
    y = model.model.endog
    e = y - (x @ model.params)
    e1 = e[:-1]; e0 = e[1:]
    rho0 = np.dot(e1,e[1:])/np.dot(e1,e1)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1 = ols_ar1(model,rho0,drop1=True)
        e = y - (x @ model1.params)
        e1 = e[:-1]; e0 = e[1:]
        rho1 = np.dot(e1,e[1:])/np.dot(e1,e1)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        print('Rho = ', rho0)
    
    