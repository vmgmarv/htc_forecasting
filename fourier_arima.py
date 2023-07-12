# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:30:48 2023

@author: Marvin Gabriel
"""

import bornly as bns
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

flights = bns.load_dataset("flights")
flights["t"] = np.arange(len(flights))
PERIOD = 12
n_steps = 12
train = flights.iloc[:-n_steps].copy()
test = flights.iloc[-n_steps:].copy()


def get_fourier_features(n_order, period, values):
    fourier_features = pd.DataFrame(
        {
            f"fourier_{func}_order_{order}": getattr(np, func)(
                2 * np.pi * values * order / period
            )
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    return fourier_features


best_aicc = None
best_n_order = None

for n_order in range(1, 8):
    train_fourier_features = get_fourier_features(n_order, PERIOD, train["t"])
    arima_exog_model = auto_arima(
        y=np.log(train["passengers"]),
        exogenous=train_fourier_features,
        seasonal=False,
    )
    if best_aicc is None or arima_exog_model.aicc() < best_aicc:
        best_aicc = arima_exog_model.aicc()
        best_norder = n_order

train_fourier_features = get_fourier_features(best_norder, PERIOD, train["t"])
arima_exog_model = auto_arima(
    y=np.log(train["passengers"]),
    exogenous=train_fourier_features,
    seasonal=False,
)
test_fourier_features = get_fourier_features(best_norder, PERIOD, test["t"])
y_arima_exog_forecast = arima_exog_model.predict(
    n_periods=n_steps,
    exogenous=test_fourier_features,
)
test["forecast"] = np.exp(y_arima_exog_forecast)
bns.lineplot(
    data=test.melt(
        id_vars=["t"], value_vars=["forecast", "passengers"], var_name="algo"
    ),
    x="t",
    y="value",
    hue="algo",
).figure