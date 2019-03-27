#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:56:55 2019

@author: serhiibahdasaryants
"""
import matplotlib
import matplotlib.pyplot as plt

for k in range(300):
    print(k)
    plt.axes(xlim=(0, Ny), ylim=(0, Nx))
    plt.scatter(y[:,k], x[:,k], alpha=0.5)
    name = '/Users/serhiibahdasaryants/ANIMATION/' + 'kapusta' + str(k) + '.jpeg'
    plt.savefig(name)
    #plt.show()
    plt.clf()