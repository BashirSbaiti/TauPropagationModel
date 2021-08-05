#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:06:50 2021

@author: Michaelkhalfin
"""

from scipy.stats import ttest_rel
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simdata.csv")
rf = list(df["rf"][df["Group"]==0])
tp = list(df["tp"][df["Group"]==0])

# check for normal distributions
fig, ax = plt.subplots()
ax.boxplot([rf,tp])
ax.set_title('Check for Normal Distributions')
ax.set_xticklabels(["Tau Reduction Factor", "Affected Tau Population"])

stat, p = ttest_rel(rf, tp)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')