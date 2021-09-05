#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:10:34 2021

@author: chloepeng
"""

import numpy as np
def forward_subs(L,b):
    y=[]
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i]=y[i]-(L[i,j]*y[j])
        y[i]=y[i]/L[i,i]

    return y


def cholupdate(R,x,sign):
    p = np.size(x)
    x = x.T
    for k in range(p):
      if sign == '+':
        r = np.sqrt(R[k,k]**2 + x[k]**2)
      elif sign == '-':
        r = np.sqrt(R[k,k]**2 - x[k]**2)
      c = r/R[k,k]
      s = x[k]/R[k,k]
      R[k,k] = r
      if sign == '+':
        R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
      elif sign == '-':
        R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
      x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R.T


  
