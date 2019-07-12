# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:00:33 2019

@author: ADMIN
"""

import numpy as np
from runGFV import run_GFV

a=np.zeros((1,14,14,512))
b=np.zeros((5,5,2))
for h in range(5):
    for w in range(5):
        b[h][w][0]=h+3
        b[h][w][1]=w+2
for h in range(14):
    for w in range(14):
        a[0][h][w][0]=h
        a[0][h][w][1]=w
fv=run_GFV(a,b)