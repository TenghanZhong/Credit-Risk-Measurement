# Import required libraries
import math
import os
import pandas as pd
import numpy as np
from scipy.optimize import fsolve  # Solve using fsolve
from scipy.optimize import root  # Solve using root
import scipy.stats as stats

# Last day closing price (Note: this comment may refer to data used later)
r = 0.03  # Pre-assigned risk-free interest rate
T = 1  # Pre-determined time period (1 year) to calculate default probability

import matplotlib.pyplot as plt
from scipy.stats import norm


def N(x):  # Define the cumulative distribution function of the normal distribution as N(x)
    return stats.norm.cdf(x)


def func(i):  # Define a system of two equations with two variables
    Sigma_Vt = i[0]
    x = i[1]
    EtoD = St / F
    d1 = ((np.log(abs(x * EtoD))) + (r + 0.5 * (Sigma_Vt ** 2) * T)) / (Sigma_Vt * np.sqrt(
        T))  # Using abs(x*EtoD) to avoid negative values during iteration that would cause log to fail
    d2 = d1 - Sigma_Vt * np.sqrt(T)
    return [
        x * N(d1) - N(d2) * math.exp(-r * T) / EtoD - 1,
        x * N(d1) * Sigma_Vt - Sigma_St
    ]


# Solve the system of two equations
# result = root(func, [1, 1]).x  # You can also solve the system using root
# print('The solution of the system is {}'.format(result))
data = pd.read_csv("C:\\Users\\zth020906\\Desktop\\非ST15.csv", encoding='gbk')
for s in range(0, 15):
    a = data['市值波动率'][s]
    b = data["流动负债合计"][s]
    c = data["非流动负债合计"][s]
    d = data["流通股"][s]
    e = data['非流通股'][s]
    f = data['每股净资产BPS'][s]
    g = data["收盘价"][s]
    aa = float(a)
    bb = float(b)
    cc = float(c)
    dd = float(d)
    ee = float(e)
    ff = float(f)
    gg = float(g)
    Sigma_St = aa
    SD = bb
    LD = cc
    tradable_share = dd
    non_tradable_share = ee
    BPS = ff
    St_close = gg
    F = SD + 0.5 * LD
    St = tradable_share * St_close + non_tradable_share * BPS
    result = 0
    result = fsolve(func, x0=[1, 1])  # Solve the system of two equations
    Sigma_Vt = result[0]
    print('The volatility of the firm’s asset value is {}'.format(Sigma_Vt))
    Vt = result[1] * St
    print('The initial asset value of the firm is {}'.format(Vt))
    DP = SD + 0.5 * LD
    print('The default point of the firm is {}'.format(DP))

    DD_Empirical = (Vt - DP * math.exp(-r * T)) / (Vt * Sigma_Vt * np.sqrt(
        T))  # Empirical Distance to Default (DD); here the return rate μ is temporarily replaced by the risk-free rate
    print('The empirical distance to default is {}'.format(DD_Empirical))

    DD_Theoretical = (np.log(Vt / F) + (r - 0.5 * (Sigma_Vt ** 2)) * T) / (
                Sigma_Vt * np.sqrt(T))  # Theoretical Distance to Default (DD)
    print('The theoretical distance to default is {}'.format(DD_Theoretical))

    EDF = norm.cdf(-DD_Theoretical)  # Calculate default probability (EDF) using the theoretical distance to default
    print(f'The theoretical default probability for this company is: {EDF}')
