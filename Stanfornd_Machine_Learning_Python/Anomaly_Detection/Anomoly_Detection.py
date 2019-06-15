#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 07:20:14 2019

@author: khaled
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from scipy.io import loadmat

data=loadmat('/mnt/407242D87242D1F8/study/anaconda/Anomaly_Detection/data/ex8data1.mat')
