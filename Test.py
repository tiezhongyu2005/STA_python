# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:35:17 2021

@author: 中南大学自动化学院  智能控制与优化决策课题组
"""

from STA_EN import STA
import Benchmark
import numpy as np
import matplotlib.pyplot as plt




funfcn = Benchmark.Quadconvex
Dim = 10
Range = np.repeat([0,2*Dim],Dim).reshape(-1,Dim)
Maxiter = 100*Dim
SE = 30
dict_opt = {'alpha': 1, 'beta': 1, 'gamma': 1, 'delta': 1}

sta = STA(funfcn,Range,Maxiter,SE,dict_opt)
sta.run()

print(sta.Best)
y = sta.history
x = np.arange(y.size).reshape(-1,1)
plt.semilogy(x,y,'b-o')
plt.xlabel('Ierations')
plt.ylabel('fitness')
plt.show()

